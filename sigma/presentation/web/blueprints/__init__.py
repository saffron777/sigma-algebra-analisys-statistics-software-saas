"""Flask blueprints: thin controllers that validate input, call one use case and serialise the output."""

from flask import Flask

from sigma.presentation.web.blueprints import (
    anova_api,
    console_api,
    datasets_api,
    descriptive_api,
    distributions_api,
    hypothesis_api,
    markets_api,
    pages,
    regression_api,
    results_api,
)

ALL_BLUEPRINTS = (
    pages.bp,
    datasets_api.bp,
    distributions_api.bp,
    descriptive_api.bp,
    hypothesis_api.bp,
    regression_api.bp,
    anova_api.bp,
    markets_api.bp,
    results_api.bp,
    console_api.bp,
)


def register_blueprints(app: Flask) -> None:
    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)
