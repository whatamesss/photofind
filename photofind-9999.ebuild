# Copyright 2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

PYTHON_COMPAT=( python3_13 python3_14 )

inherit git-r3 python-single-r1 desktop xdg

DESCRIPTION="Local AI-powered photo search and organizer using CLIP and PyQt6"
HOMEPAGE="https://github.com/whatamesss/photofind"
EGIT_REPO_URI="https://github.com/whatamesss/photofind.git"

LICENSE="GPL-3+"
SLOT="0"
KEYWORDS=""
IUSE="jdupes"

REQUIRED_USE="${PYTHON_REQUIRED_USE}"

RDEPEND="
	${PYTHON_DEPS}
	dev-python/numpy
	dev-python/pillow
	dev-python/pyqt6[gui,widgets]
	dev-python/tqdm
	sci-ml/pytorch[${PYTHON_SINGLE_USEDEP}]
	sci-ml/transformers[torch]
	x11-misc/xdg-utils
	jdupes? ( app-misc/jdupes )
"

pkg_setup() {
	python-single-r1_pkg_setup
}

src_prepare() {
	default

	python_fix_shebang .
}

src_install() {
	newbin photofind.py photofind

	insinto /usr/share/icons/hicolor/256x256/apps
	doins photofind.png

	make_desktop_entry \
		photofind \
		PhotoFind \
		photofind \
		"Graphics;Photography;"

	einstalldocs
}

pkg_postinst() {
	xdg_icon_cache_update

	if [[ -z ${REPLACING_VERSIONS} ]]; then
		elog "PhotoFind has been installed."
		elog "The first time you run PhotoFind, it will download"
		elog "the 'openai/clip-vit-large-patch14' model."

		if ! use jdupes; then
			elog "Duplicate detection is disabled because the"
			elog "'jdupes' USE flag is not enabled."
		fi
	fi
}

pkg_postrm() {
	xdg_icon_cache_update
}
