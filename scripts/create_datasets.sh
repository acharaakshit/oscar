projectroot=${PROJECTDIR}
# PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset HCP --FS 3T
PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset ADNI --FS 3T
PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset ADNI --FS 1.5T
# PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset OASIS --FS 3T
# PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset OASIS --FS 1.5T
# PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset A4 --FS 3T
# PYTHONPATH=$projectroot python3 -u $projectroot/process/create_data_objects.py --dataset UKBB --FS 3T