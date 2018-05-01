import datetime


ROOT_LOG_DIR = 'tf_logs'
TIME_FORMAT = '%Y%m%d_%H%M%S'


def get_logdir(chapter_number: int):
    now = datetime.datetime.now().strftime(TIME_FORMAT)
    return f'../{ROOT_LOG_DIR}/chapter_{chapter_number:02}/run_{now}/'
