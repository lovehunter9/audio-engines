def build_app(supports):
    from ..s2st.server import build_app as build

    return build(supports)


def run(supports):
    from ..s2st.server import main

    main(supports)
