import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "mlProject"



list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion_component.py",
    f"src/{project_name}/components/data_transformation_component.py",
    f"src/{project_name}/components/data_validation_component.py",
    f"src/{project_name}/components/model_evaluation_component.py",
    f"src/{project_name}/components/model_trainer_component.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/utils.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/step_01_data_ingestion_pipeline.py",
    f"src/{project_name}/pipeline/step_02_data_validation_pipeline.py",
    f"src/{project_name}/pipeline/step_03_data_transformation_pipeline.py",
    f"src/{project_name}/pipeline/step_04_model_trainer_pipeline.py",
    f"src/{project_name}/pipeline/step_05_model_evaluation_pipeline.py",
    f"src/{project_name}/pipeline/step_06_prediction_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "test/__init__.py",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "tox.ini",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "research/trials.ipynb",
    "templates/index.html",
    "pyproject.toml",
    "init_setup.sh"


]



for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")