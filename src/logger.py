import json
import os
import sys
from logging import Logger, config, getLogger

# 実行モードの切り替え
# テンプレートではデフォルトデバッグモードに設定
VALIDATION_MODE = os.getenv("VALIDATION_MODE", "debug").lower()

# loggerのconfig
DEV_CONFIG_FILE = "./log_config/dev_log_config.json"
PROD_CONFIG_FILE = "./log_config/prod_log_config.json"


class SingletonLogger:
    """シングルトンパターンを用いたロガークラス。

    このクラスを通じて取得したロガーは、アプリケーション全体で共有される。

    使用例:
        logger = SingletonLogger().get_logger(__name__)

    """

    _instance = None

    def __new__(cls, *, log_dir: str | None = None) -> "SingletonLogger":
        if not cls._instance:
            cls._set_config(log_dir)
            cls._instance = super(SingletonLogger, cls).__new__(cls)
        return cls._instance

    def get_logger(self, name: str) -> Logger:
        return getLogger(name)

    @classmethod
    def set_log_dir(cls, log_dir: str) -> None:
        """ロギングの出力先を変更する。

        Args:
            log_dir (str): ロギングの出力先

        """
        cls._set_config(log_dir)

    @classmethod
    def _set_config(cls, log_dir: str | None) -> None:
        """loggerのconfig設定を行う。

        Args:
            log_dir (str): ロギングの出力先

        """
        # Pyinstaller で exe 化した場合を考慮
        bundle_dir = getattr(sys, "_MEIPASS", os.path.abspath(os.getcwd()))
        config_file = (
            PROD_CONFIG_FILE if VALIDATION_MODE == "prod" else DEV_CONFIG_FILE
        )

        with open(os.path.join(bundle_dir, config_file), "r") as f:
            log_conf = json.load(f)

        if log_dir is not None:
            log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(
                log_dir, log_conf["handlers"]["fileHandler"]["filename"]
            )

        config.dictConfig(log_conf)
