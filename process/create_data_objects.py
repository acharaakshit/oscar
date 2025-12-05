import nibabel as nib
import os
import numpy as np
from tqdm import tqdm
import argparse
import nibabel as nib
import json
import pandas as pd
import scipy.ndimage as ndi
from nilearn.image import crop_img
from collections.abc import Sequence
import yaml
import SimpleITK as sitk

def perform_N4(img_path):
    sitkimage = sitk.ReadImage(img_path)
    cast_back = False
    if sitkimage.GetPixelIDTypeAsString() == '16-bit signed integer': # for IXI dataset
        sitkimage = sitk.Cast(sitkimage, sitk.sitkFloat32)
        cast_back = True
    mask = sitk.NotEqual(sitkimage, 0)
    shrinkfactor = 4 
    shrunkimage = sitk.Shrink(sitkimage, [shrinkfactor] * sitkimage.GetDimension())
    maskimage = sitk.Shrink(mask, [shrinkfactor] * sitkimage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([100, 100, 50])
    corrector.SetConvergenceThreshold(0.00001)
    corrector.SetNumberOfControlPoints(5)
    corrector.SetSplineOrder(4)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.Execute(shrunkimage, maskimage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitkimage)
    corrected_image_full_resolution = sitkimage / sitk.Exp(log_bias_field)
    corrected_image_full_resolution.CopyInformation(sitkimage)
    if cast_back:
        corrected_image_full_resolution = sitk.Cast(corrected_image_full_resolution, sitk.sitkUInt16)

    sitkimage_path = img_path.replace(".nii.gz", "_N4.nii.gz")
    sitk.WriteImage(corrected_image_full_resolution, sitkimage_path)
    return sitkimage_path


def preprocess_nifti(
        img_path: str, 
        target_shape: Sequence[int] = (256, 256, 256),
        N4: bool = False,
        ):
    if N4:
        img_path = perform_N4(img_path)
    
    img = nib.load(img_path)
    data = img.get_fdata()

    # apply nilearn cropping if image is larger than target
    cropped_img = crop_img(img, copy_header=True) if any(o > t for o, t in zip(data.shape, target_shape)) else img
    cropped_data = cropped_img.get_fdata()

    # compute padding (only if needed)
    cropped_shape = cropped_data.shape
    pad_widths = [(max(0, (t - c) // 2), max(0, (t - c + 1) // 2)) for c, t in zip(cropped_shape, target_shape)]

    # apply padding if necessary
    processed_data = np.pad(cropped_data, pad_widths, mode='constant', constant_values=0) \
        if any(c < t for c, t in zip(cropped_shape, target_shape)) else cropped_data

    # minimal resizing to avoid artifacts
    if processed_data.shape != target_shape:
        zoom_factors = [t / float(s) for s, t in zip(processed_data.shape, target_shape)]
        processed_data = ndi.zoom(processed_data, zoom_factors)

    assert processed_data.shape == target_shape

    processed_data = nib.Nifti1Image(processed_data, img.affine)
    return processed_data

def save_data(mri_data: str):
    print(f"Saving {len(mri_data)} scans!")
    # save each item separately
    for k, image in tqdm(mri_data.items()):
        nib.save(image, k)


def create_dataset(args):
    PROJECT_ROOT = os.getenv('PROJECTDIR')
    PREFIX = os.getenv('PREFIX')

    # read folders yaml file
    with open(f'{PROJECT_ROOT}/config/folder.yaml') as f:
        folders = yaml.safe_load(f)
    
    PROCESSED_FOLDER = folders['processed']
    PREPROCESSED_FOLDER = folders['preprocessed']
    dataset = args.dataset
    FS = args.FS

    data_setting = f'{dataset}_{FS}'
    print(f'Data setting is: {data_setting}')

    # create processed directories
    processed_dir = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER)
    os.makedirs(processed_dir, exist_ok=True)

    if args.dataset == 'ADNI':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        scan_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('Brain.nii.gz'): # harcoded for skull stripped images
                    scan_files.append(os.path.join(dirpath, file))

        # image ids for each scan
        image_ids = [nifti.split('/')[-1].split('_')[0] for nifti in scan_files]
        assert len(set(image_ids)) == len(image_ids) # check that only unique scans are selected
        assert len(image_ids) == len(scan_files)
        
        mri_data = {}

        for scan_file, image_id in tqdm(zip(scan_files, image_ids)):
            save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{image_id}.nii.gz')
            if os.path.isfile(save_path):
                print(f"File {save_path} already saved!")
                continue
            scan = preprocess_nifti(scan_file)
            mri_data[save_path] = scan
            if len(mri_data) >= 50:
                save_data(mri_data=mri_data)
                mri_data = {}
    elif args.dataset == 'UKBB':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        scan_files = os.listdir(root_dir)
        scan_files = [f for f in scan_files if '_Brain.nii.gz' in f]

        # only reading available data
        metadata_path = os.path.join(PREFIX, data_setting,'FILTERED.csv') # see ukbb_filter script for the same
        subjects = pd.read_csv(metadata_path)['subject'].tolist()
        subjects = [str(int(s)) for s in subjects]

        mri_data = {}
        for f in tqdm(scan_files):
            if os.path.isfile(os.path.join(root_dir, f)) and f.split(".")[0].split("_")[0] in subjects:
                save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{f.split(".")[0].split("_")[0]}.nii.gz')
                if os.path.isfile(save_path):
                    print(f"File {save_path} already saved!")
                    continue
                scan_file = os.path.join(root_dir, f)
                try:
                    scan = preprocess_nifti(scan_file, N4=True)
                except EOFError:
                    continue
                mri_data[save_path] = scan
                
                if len(mri_data) >= 50:
                    save_data(mri_data=mri_data)
                    mri_data = {}
    elif args.dataset == 'HCP':
        root_dir = os.path.join(PREFIX, args.dataset)
        directory = os.listdir(root_dir)

        mri_data = {}
        for f in tqdm(directory):
            if os.path.isdir(os.path.join(root_dir, f)) and len(f) == 6:
                subject_folder = os.path.join(root_dir, f)
                save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER ,f'{subject_folder.split("/")[-1]}.nii.gz')
                if os.path.isfile(save_path):
                    print(f"File {save_path} already saved!")
                    continue
                scan_file = os.path.join(subject_folder, 'T1w', 'T1w_acpc_dc_restore_brain.nii.gz')
                scan = preprocess_nifti(scan_file)
                mri_data[save_path] = scan
                
                if len(mri_data) >= 50:
                    save_data(mri_data=mri_data)
                    mri_data = {}
    elif args.dataset == 'A4':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        directory = os.listdir(root_dir)
        directory = [f for f in directory if 'MNI_Brain.nii.gz' in f]
        json_directory = os.listdir(root_dir)
        json_directory = [f.split('.')[0] for f in json_directory if '.json' in f]

        mri_data = {}
        unique_ids = set()
        for f in tqdm(directory):
            image_id = f.split('_')[0] + str(f.split('_')[3]) + str(f.split('_')[4])
            unique_ids.add(image_id)

        print(f"The number of unique subjects are {len(unique_ids)}")

        for f in tqdm(directory):
            image_id = f.split('_')[3]
            scan_idx = str(f.split('_')[0])
            visit_code = str(f.split('_')[4])
            main_idx = scan_idx + '_' + image_id + '_' + visit_code
            # print(main_idx, image_id, visit_code)
            corrupt_T1s = ['LEARN_B69617870_066']
            if main_idx in corrupt_T1s:
                print(f"skipping {main_idx} due to invalid input")
                continue

            if '_'.join(f.split('_')[:5]) in json_directory: #and visit_code in visit_map[scan_idx]:
                save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{main_idx}.nii.gz')
                if os.path.isfile(save_path):
                    print(f"File {save_path} already saved!")
                    continue
                scan_file = os.path.join(root_dir, f)
                json_file = os.path.join(root_dir, '_'.join(f.split('_')[:5]) + '.json')
                with open(json_file, 'r') as handle:
                    fields = json.load(handle)
                if str(fields['MagneticFieldStrength']) != str(data_setting.split('_')[-1].replace('T', '')):
                    print(f"The file {f} is not a 3T scan")
                    continue

                scan = preprocess_nifti(scan_file, N4=True)
                mri_data[save_path] = scan
                
                if len(mri_data) >= 50:
                    save_data(mri_data=mri_data)
                    mri_data = {}

    elif args.dataset == 'OASIS':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        directory = os.listdir(root_dir)
        json_directory = [f.split('.')[0].replace('_MNI_Brain', '') for f in directory if '.json' in f]
        directory = [f for f in directory if 'MNI_Brain.nii.gz' in f]

        mri_data = {}
        unique_ids = set()
        for f in tqdm(directory):
            image_id = f.split('.')[0]
            unique_ids.add(image_id)

        print(f"The number of unique subjects are {len(unique_ids)}")

        for f in tqdm(directory):
            image_id = f.split('.')[0].replace('_MNI_Brain', '')
            # use only one scan per subject for now
            if ".nii.gz" in f:
                    save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{image_id}.nii.gz')
                    if os.path.isfile(save_path):
                        print(f"File {save_path} already saved!")
                        continue
                    try:
                        assert image_id in json_directory

                        json_file = os.path.join(root_dir, image_id + '.json')

                        with open(json_file, 'r') as handle:
                            fields = json.load(handle)

                        if str(fields['MagneticFieldStrength']) != str(data_setting.split('_')[-1].replace('T', '')):
                            print(f"The file {f} is not a {str(data_setting.split('_')[-1].replace('T', ''))}T scan")
                            continue
                    except AssertionError:
                        print("Can not determine field strength, adding to 3T")

                    scan_file = os.path.join(root_dir, f)
                    scan = preprocess_nifti(scan_file)
                    mri_data[save_path] = scan
                    
                    if len(mri_data) >= 50:
                        save_data(mri_data=mri_data)
                        mri_data = {}
    elif dataset == 'ABIDE':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        directory = os.listdir(root_dir)
        scan_files = [f for f in directory if 'Brain.nii.gz' in f]
        image_ids = [nifti.split('/')[-1].split('_')[0] for nifti in scan_files]
        assert len(set(image_ids)) == len(image_ids) # check that only unique scans are selected
        assert len(image_ids) == len(scan_files)
        mri_data = {}
        for image_id, f in tqdm(zip(image_ids, scan_files)):
            save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{image_id}.nii.gz')
            if os.path.isfile(save_path):
                print(f"File {save_path} already saved!")
                continue
            
            scan_file = os.path.join(root_dir, f)
            scan = preprocess_nifti(scan_file, N4=True)
            mri_data[save_path] = scan
            
            if len(mri_data) >= 50:
                save_data(mri_data=mri_data)
                mri_data = {}
    elif dataset == 'IXI':
        root_dir = os.path.join(PREFIX, data_setting, PREPROCESSED_FOLDER)
        directory = os.listdir(root_dir)
        directory = [f for f in directory if 'Brain.nii.gz' in f]
        mri_data = {}
        for f in tqdm(directory):
            image_id = f.split('-')[0].replace('IXI', '')
            save_path = os.path.join(PREFIX, data_setting, PROCESSED_FOLDER, f'{image_id}.nii.gz')
            if os.path.isfile(save_path):
                print(f"File {save_path} already saved!")
                continue
            
            scan_file = os.path.join(root_dir, f)
            scan = preprocess_nifti(scan_file, N4=True)
            mri_data[save_path] = scan
            
            if len(mri_data) >= 50:
                save_data(mri_data=mri_data)
                mri_data = {}
    else:
        raise ValueError('Incorrect value for dataset passed')

    print(f"Save the remaining {len(mri_data)} MRI Scans")
    save_data(mri_data=mri_data)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This is a data object creation script")
    
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--FS", type=str, default="3T", help='Field Strength')

    # Parse the arguments
    args = parser.parse_args()
    # Call main with parsed arguments
    create_dataset(args)
