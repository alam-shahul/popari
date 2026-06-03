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
    targets = session.posargs or ["popari", "tests"]
    session.run(
        "flake8",
        *targets,
        "--filename",
        "*.py",
        "--exclude",
        "popari/genes_ncbi_mus_musculus_proteincoding.py",
    )


@nox.session
def lint(session):
    targets = (flake8,)
    for t in targets:
        session.log(f"Running {t.__name__}")
        t(session)


@nox.session
def unittests(session):
    session.run_install(
        "uv",
        "sync",
        "--group",
        "test",
        "--group",
        "dev",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest")


nox.options.sessions = [
    "lint",
    # "unittests",
]
