# Copyright 2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

PYTHON_COMPAT=( python3_13 )
DISTUTILS_USE_PEP517=setuptools
inherit python-single-r1 desktop

DESCRIPTION="Local AI-powered photo search and organizer using CLIP and PyQt6"
HOMEPAGE="https://github.com/whatamesss/photofind"
SRC_URI="https://github.com/whatamesss/photofind/archive/refs/tags/v${PV}.tar.gz"

LICENSE="GPL-3+"
SLOT="0"
KEYWORDS="~amd64 ~x86"
IUSE="jdupes"  # CUDA flag removed, managed manually via package.use

REQUIRED_USE="${PYTHON_REQUIRED_USE}"

RDEPEND="${PYTHON_DEPS}
    dev-python/PyQt6[${PYTHON_SINGLE_USEDEP},gui,widgets]
    dev-python/pillow[${PYTHON_SINGLE_USEDEP}]
    dev-python/transformers[${PYTHON_SINGLE_USEDEP}]
    dev-python/numpy[${PYTHON_SINGLE_USEDEP}]
    dev-python/tqdm[${PYTHON_SINGLE_USEDEP}]

    # PyTorch and Caffe2 flags are managed globally/locally in package.use
    # to ensure proper propagation of CUDA support.
    sci-libs/pytorch[${PYTHON_SINGLE_USEDEP},python]
    sci-ml/caffe2

    xdg-utils
    jdupes? ( app-misc/jdupes )
"

BDEPEND=""

S="${WORKDIR}/${PN}-${PV}"

src_install() {
    python_newscript photofind.py photofind
    make_desktop_entry "/usr/bin/photofind" "PhotoFind" "" "Graphics;Photography;"
    einstalldocs
}

pkg_postinst() {
    if [[ ! ${REPLACING_VERSIONS} ]]; then
        elog "PhotoFind has been installed."
        elog "On first run, the application will download the 'openai/clip-vit-large-patch14' model."
        if ! use jdupes; then
            elog "The 'jdupes' USE flag is disabled. Duplicate detection is disabled."
        fi
    fi
}
