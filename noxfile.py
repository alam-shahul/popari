import nox


@nox.session
def flake8(session):
    session.install(
        "flake8",
        "flake8-absolute-import",
        "flake8-bugbear",
        "flake8-builtins",
        "flake8-colors",
        "flake8-commas",
        "flake8-comprehensions",
        # "flake8-docstrings",
        "flake8-pyproject",
        "flake8-use-fstring",
        "pep8-naming",
    )
    session.run("flake8", "--filename", "*.py", "--exclude", "popari/genes_ncbi_mus_musculus_proteincoding.py")


@nox.session
def lint(session):
    targets = (flake8,)
    for t in targets:
        session.log(f"Running {t.__name__}")
        t(session)


@nox.session
def unittests(session):
    session.install(".[test]")
    session.run("pytest")


nox.options.sessions = [
    "lint",
    # "unittests",
]
