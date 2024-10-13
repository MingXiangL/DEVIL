import os

from pydantic_settings import BaseSettings

from geminiplayground.utils import LibUtils


class Settings(BaseSettings):
    database_url: str
    echo_sql: bool = True


playground_home = LibUtils.get_lib_home()
database_file = os.path.join(playground_home, "data.db")
database_uri = f"sqlite+aiosqlite:///{database_file}"
settings = Settings(database_url=database_uri, echo_sql=False)
