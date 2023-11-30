import logging

from rich.logging import RichHandler

# import logging
# from rich.logging import RichHandler


def get_stroke_logger(name="rich", level="NOTSET"):

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    return logging.getLogger(name)


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


if __name__ == "__main__":

    log = logging.getLogger("rich")
    log.info("Hello, World!")
