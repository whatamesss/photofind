# Copyright 2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

PYTHON_COMPAT=( python3_13 )
DISTUTILS_USE_PEP517=setuptools
inherit git-r3 python-single-r1 desktop

DESCRIPTION="Local AI-powered photo search and organizer using CLIP and PyQt6"
HOMEPAGE="https://github.com/whatamesss/photofind"

EGIT_REPO_URI="https://github.com/whatamesss/photofind.git"

LICENSE="GPL-3+"
SLOT="0"
KEYWORDS=""
IUSE="jdupes"

REQUIRED_USE="${PYTHON_REQUIRED_USE}"

RDEPEND="${PYTHON_DEPS}
    dev-python/pyqt6[gui,widgets]
    dev-python/pillow
    sci-ml/transformers[torch]
    dev-python/numpy
    dev-python/tqdm
    sci-ml/pytorch[${PYTHON_SINGLE_USEDEP}]
    sci-ml/caffe2
    x11-misc/xdg-utils
    jdupes? ( app-misc/jdupes )
"

BDEPEND=""

S="${WORKDIR}/${P}"

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
