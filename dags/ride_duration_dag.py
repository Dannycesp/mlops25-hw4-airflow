from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='ride_duration_batch_pipeline',
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=True,
    tags=['batch', 'zoomcamp'],
) as dag:

    download_data = DockerOperator(
        task_id='download_data',
        image='ride-duration-batch',
        command='--step download --year {{ execution_date.year }} --month {{ execution_date.month }}',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {
                'source': '/mnt/Disco_F_mnt/Documents_F/Courses/DataTalksClub/MLOPS_ZOOMCAMP/mlops25-hw4-airflow/output',
                'target': '/data/output',
                'type': 'bind'
            }
        ],
        auto_remove=True,
    )

    predict_duration = DockerOperator(
        task_id='predict_duration',
        image='ride-duration-batch',
        command='--step predict --year {{ execution_date.year }} --month {{ execution_date.month }}',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {
                'source': '/mnt/Disco_F_mnt/Documents_F/Courses/DataTalksClub/MLOPS_ZOOMCAMP/mlops25-hw4-airflow/output',
                'target': '/data/output',
                'type': 'bind'
            }
        ],
        auto_remove=True,
    )

    save_results = DockerOperator(
        task_id='save_results',
        image='ride-duration-batch',
        command='--step save --year {{ execution_date.year }} --month {{ execution_date.month }}',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            {
                'source': '/mnt/Disco_F_mnt/Documents_F/Courses/DataTalksClub/MLOPS_ZOOMCAMP/mlops25-hw4-airflow/output',
                'target': '/data/output',
                'type': 'bind'
            }
        ],
        auto_remove=True,
    )

    download_data >> predict_duration >> save_results