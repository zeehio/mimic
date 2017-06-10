#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

BUILDFROM="$PWD"

if [ "x${PKG_CONFIG}" = "x" ]; then
  PKG_CONFIG=`which pkg-config`
fi

which "$PKG_CONFIG" ||(echo "Please install pkg-config"; exit 1)

"$PKG_CONFIG" --exists libpcre2-8 && HAVE_PCRE2="yes" || HAVE_PCRE2="no"
echo "Have PCRE2? [${HAVE_PCRE2}]"

if [ "x${HAVE_PCRE2}" = "xno" ]; then
    mkdir -p "${BUILDFROM}/build_thirdparty/pcre2" || exit 1
    cd "${BUILDFROM}/build_thirdparty/pcre2" || exit 1
    ${SCRIPTPATH}/thirdparty/pcre2/configure --disable-option-checking $@ || exit 1
    make || exit 1
    make install || exit 1
fi

cd "${BUILDFROM}"

