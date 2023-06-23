import os
from pathlib import Path


project_name = 'loanClassifier'

list_of_files = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/components/data_ingestion.py',
    f'src/{project_name}/components/data_transformation.py',
    f'src/{project_name}/components/model_trainer.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/pipeline/training_pipeline.py',
    f'src/{project_name}/pipeline/prediction_pipeline.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/logger.py',
    f'src/{project_name}/exception.py',
    'app.py',
    'main.py',
    'requirements.txt',
    'setup.py',
    'notebooks/data',
    'templates/index.html',
    'Dokerfile'
    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
