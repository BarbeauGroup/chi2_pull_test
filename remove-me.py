from flux.ssb_pdf import make_ssb_pdf
from utils.loadparams import load_params


def test_make_ssb_pdf():

    params = load_params("config/csi.json")

    ssb_pdf = make_ssb_pdf(params)

if __name__ == "__main__":
    test_make_ssb_pdf()


