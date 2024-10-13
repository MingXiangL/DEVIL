import logging

from diskcache import Cache
from geminiplayground.utils import LibUtils

logger = logging.getLogger("rich")

playground_home = LibUtils.get_lib_home()
cache_folder = playground_home.joinpath(".cache").resolve()
logger.info(f"Using cache directory: {cache_folder}")
cache = Cache(directory=str(cache_folder))
