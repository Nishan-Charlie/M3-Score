import m3score
from m3score import protocol


def test_version_is_exposed():
    assert m3score.__version__ == "0.1.0"


def test_protocol_constants_are_the_published_ones():
    assert protocol.PROTOCOL_VERSION == "m3-v1"
    assert protocol.BACKBONE_ID == "Snarcy/RadioDino-s16"
    assert protocol.BACKBONE_REVISION == "773d28d873455153495d39179fc42ea816aa1212"
    assert protocol.IMAGE_SIZE == 224
    assert protocol.LAYERS == (12, 9, 4)
    assert (protocol.FIDELITY_LAYER, protocol.MEMORIZATION_LAYER, protocol.COVERAGE_LAYER) == (
        12,
        9,
        4,
    )
    assert protocol.RBF_BANDWIDTH_SCALES == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert protocol.COVERAGE_K == 5
    assert protocol.MIN_REAL_IMAGES == 6


def test_errors_and_warnings_have_the_documented_bases():
    assert issubclass(m3score.M3ScoreWarning, UserWarning)
    assert issubclass(m3score.M3ScoreError, RuntimeError)
