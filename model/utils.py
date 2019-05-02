"""General utility functions"""
# json => javascript object notation
import json
import logging




class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    # 클래스 함수의 첫번째 인수로 self를 써 줘야만 해당 함수를 인스턴스의 함수로 사용 가능
    # __init__ 같은 경우 초기화 메서드 즉, 함수의 인자들을 미리 선언해주는 역할
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file""" # save 이기 때문에 open( , w)
        with open(json_path, 'w') as f:
            # dump => 문자열을 한 줄로 길게 표현되게 만들어줌, 읽기 힘드니까 indent=4 를 써줌
            # __dict__ 는 모듈의 심볼 테이블을 포함하고있는 딕션너리
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    ## 로거 설명
    # https://hamait.tistory.com/880

    # getLogger() => 자신만의 로거를 만들기
    logger = logging.getLogger()
    # setLevel() => 로거 레벨 설정 여기서는 info 이상만 출력 하도록 함
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        # 파일 핸들러는 내가 로킹한 정보가 출력되는 파일 위치 설정
        file_handler = logging.FileHandler(log_path)
        # 포메팅은 기타 정보를 같이 출력하고싶을떄 사용
        # asctime = 시간, levelname = 로깅레벨, message = 메세지, name = 로거이름
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console // 파일말고 콘솔을 통해 출력 하도록 하는것
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)











