import os
import typer
from typing_extensions import Annotated

cli = typer.Typer(invoke_without_command=True)


def check_api_key():
    """
    Check if the api key is set
    """
    # attempt to load the api key from the .env file
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    # receive the api key from the command line
    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not api_key:
        typer.echo(
            "Please set the AISTUDIO_API_KEY environment variable, or create a .env file with the api key obtained "
            "from https://aistudio.google.com/app/apikey"
        )
        raise typer.Abort()


@cli.command()
def ui(
        host: str = "localhost",
        port: int = 8081,
        workers: int = os.cpu_count() * 2 + 1,
        reload: Annotated[bool, typer.Option("--reload")] = False,
        api_key: str = typer.Option(None, envvar="GOOGLE_API_KEY")
):
    """
    Launch the web app
    """
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    check_api_key()

    import uvicorn

    uvicorn.run(
        "geminiplayground.web.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@cli.command()
def api(
        host: str = "localhost",
        port: int = 8081,
        workers: int = os.cpu_count() * 2 + 1,
        reload: Annotated[bool, typer.Option("--reload")] = True,
        api_key: str = typer.Option(None, envvar="GOOGLE_API_KEY")
):
    """
    Launch the API
    """
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    check_api_key()

    import uvicorn

    uvicorn.run(
        "geminiplayground.web.api:api",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


def run():
    """
    Run the app
    """
    cli()


if __name__ == "__main__":
    run()
