project('mimic', 'c', default_options : ['c_std=c99'], version: '1.2.0.2')

mimic_core_deps = []
c_compiler = meson.get_compiler('c')
host_system = host_machine.system()


conf_data = configuration_data()

pkgdatadir = join_paths(get_option('prefix'), get_option('datadir'), 'ttsmimic')
conf_data.set('PKGDATADIR', '"' + pkgdatadir + '"')


# OpenMP support (only needed for multithreaded test)
has_openmp = c_compiler.has_argument('-fopenmp')

use_openmp = get_option('openmp')
if not has_openmp
  use_openmp = false
endif
message('Openmp support: @0@'.format(use_openmp))

mmap_type_none = 0
mmap_type_posix = 1
mmap_type_win32 = 2

conf_data.set('MMAP_TYPE_NONE', mmap_type_none)
conf_data.set('MMAP_TYPE_POSIX', mmap_type_posix)
conf_data.set('MMAP_TYPE_WIN32', mmap_type_win32)

has_posix_mmap = c_compiler.has_function('mmap')

if host_system == 'windows'
  conf_data.set('MMAP_TYPE', mmap_type_win32)
elif has_posix_mmap
  conf_data.set('MMAP_TYPE', mmap_type_posix)
else
  conf_data.set('MMAP_TYPE', mmap_type_none)
endif


check_headers = [
  ['HAVE_SYS_SOCKET_H', 'sys/socket.h']
]

foreach h : check_headers
  if c_compiler.has_header(h.get(1))
    conf_data.set(h.get(0), 1)
  endif
endforeach


# Check HTS Library:
use_hts = get_option('enable-hts')
use_hts_bundled = false
if use_hts
  conf_data.set('HAVE_HTSENGINE', 1)
  # Check if we need the bundled hts: (a proper meson dependency for hts would be better)
  use_hts_bundled = not c_compiler.has_header('HTS_engine.h')
  if use_hts_bundled
    message('HTSEngine was not detected, using bundled HTSEngine')
  else
    hts_dep = declare_dependency(link_with : 'HTSEngine')
    mimic_core_deps += hts_dep
  endif
endif



# PCRE-2
pcre2_dep = dependency('libpcre2-8')
mimic_core_deps += pcre2_dep

# Audio driver:

selected_audio = get_option('audio')
selected_audio_is_auto = selected_audio == 'auto'
if selected_audio == 'auto'
  if host_system == 'windows' or host_system == 'cygwin' or host_system == 'darwin'
    selected_audio = ['portaudio', 'none']
  elif host_system == 'linux'
    selected_audio = ['alsa', 'pulseaudio', 'none']
  elif host_system == 'bsd'
    selected_audio = ['oss', 'none']
  elif host_system == 'android'
    selected_audio = ['none']
  else
    selected_audio = ['none']
  endif
else
  selected_audio = [selected_audio]
endif

have_audio = false
final_audio = 'none'
foreach audio : selected_audio
  if not have_audio
    if audio == 'alsa'
      alsa_dep = dependency('alsa', version: '>= 1.0.11', required: false)
      if alsa_dep.found()
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_ALSA', '')
        mimic_core_deps += alsa_dep
      endif
    elif audio == 'portaudio'
      portaudio_dep = dependency('portaudio-2.0', version: '>= 19', required: false)
      if portaudio_dep.found()
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_PORTAUDIO', '')
        mimic_core_deps += portaudio_dep
      endif
    elif audio == 'none'
      have_audio = true
      final_audio = 'none'
    elif audio == 'pulseaudio'
      pulseaudio_dep = dependency('libpulse', required: false)
      if pulseaudio_dep.found()
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_PULSEAUDIO', '')
        mimic_core_deps += pulseaudio_dep
      endif
    elif audio == 'sun'
      if c_compiler.has_header('sys/audioio.h')
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_SUNOS', '')
      endif
    elif audio == 'oss'
      if c_compiler.has_header('machine/soundcard.h')
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_FREEBSD', '')
      endif
    elif audio == 'wince'
      if c_compiler.has_header('mmsystem.h')
        have_audio = true
        final_audio = audio
        conf_data.set('CST_AUDIO_WINCE', '')
        winmm_dep = declare_dependency(link_with : 'winmm')
        mimic_core_deps += winmm_dep
      endif
    endif
  endif
endforeach

if selected_audio_is_auto and final_audio == 'none'
   message('Warning: no audio module was found.')
endif

use_gpl = get_option('gpl')
message('GPL allowed: @0@'.format(use_gpl))

# Saga support:
saga_dep = dependency('saga', required: false)
have_saga = saga_dep.found()
if have_saga
  if use_gpl
    mimic_core_deps += saga_dep
  else
    message('Saga usage disabled, due to non-gpl build')
  endif
endif


use_plugins = get_option('plugins')
message('Use plugin system: @0@'.format(use_plugins))

if use_plugins
  dl_dep = c_compiler.find_library('dl', required: false) # FIXME: Handle non-POSIX case
  if dl_dep.found()
    conf_data.set('MIMIC_USE_PLUGINS', 1)
    mimic_core_deps += dl_dep
  endif
endif

# libttsmimic:
libttsmimic_include_dirs = []
libttsmimic_SOURCES = []



###### src/audio ###############
src_audio_SOURCES = [
  'src/audio/au_alsa.c',
  'src/audio/auclient.c',
  'src/audio/au_command.c',
  'src/audio/audio.c',
  'src/audio/au_none.c',
  'src/audio/au_oss.c',
  'src/audio/au_portaudio.c',
  'src/audio/au_pulseaudio.c',
  'src/audio/auserver.c',
  'src/audio/au_streaming.c',
  'src/audio/au_sun.c',
  'src/audio/au_wince.c']
libttsmimic_SOURCES += src_audio_SOURCES

###### src/cg ##################
src_cg_SOURCES = [
  'src/cg/cst_cg.c',
  'src/cg/cst_mlsa.c',
  'src/cg/cst_mlpg.c',
  'src/cg/cst_vc.c',
  'src/cg/cst_cg_load_voice.c',
  'src/cg/cst_cg_dump_voice.c',
  'src/cg/cst_cg_map.c',
  'src/cg/cst_spamf0.c']
libttsmimic_SOURCES += src_cg_SOURCES

####### src/hrg ###############
src_hrg_SOURCES = [
  'src/hrg/cst_utterance.c',
  'src/hrg/cst_relation.c',
  'src/hrg/cst_item.c',
  'src/hrg/cst_ffeature.c',
  'src/hrg/cst_rel_io.c']
libttsmimic_SOURCES += src_hrg_SOURCES

####### src/hts ###############
src_hts_SOURCES = [
  'src/hts/flite_hts_engine.c']

if use_hts_bundled
  libttsmimic_include_dirs += include_directories('src/hts/hts_engine_API/include')
  install_headers('src/hts/hts_engine_API/include/HTS_engine.h', subdir: 'ttsmimic')
  src_hts_SOURCES += [
    'src/hts/hts_engine_API/lib/HTS_audio.c',
    'src/hts/hts_engine_API/lib/HTS_engine.c',
    'src/hts/hts_engine_API/lib/HTS_misc.c',
    'src/hts/hts_engine_API/lib/HTS_pstream.c',
    'src/hts/hts_engine_API/lib/HTS_sstream.c',
    'src/hts/hts_engine_API/lib/HTS_model.c',
    'src/hts/hts_engine_API/lib/HTS_vocoder.c',
    'src/hts/hts_engine_API/lib/HTS_gstream.c',
    'src/hts/hts_engine_API/lib/HTS_label.c']
endif

libttsmimic_SOURCES += src_hts_SOURCES


###### src/lexicon #########

src_lexicon_SOURCES = [
  'src/lexicon/cst_lexicon.c',
  'src/lexicon/cst_lts.c',
  'src/lexicon/cst_lts_rewrites.c']

libttsmimic_SOURCES += src_lexicon_SOURCES

###### src/regex ########

create_mimic_regex_bin = executable('create_mimic_regex',
                                    ['src/regex/compile_mimic_regex.c',
                                     'src/regex/cst_regex.c', 'src/regex/regexp.c',
                                     'src/regex/regsub.c'],
                                    include_directories: include_directories('include'),
                                    native: true)

gen_src = custom_target('create_cst_regex_defs.h',
                        input : [],
                        output : ['cst_regex_defs.h'],
                        command : [create_mimic_regex_bin, '@OUTPUT0@'])



src_regex_SOURCES = [
  'src/regex/cst_regex.c',
  'src/regex/regexp.c',
  'src/regex/regsub.c', gen_src]

libttsmimic_SOURCES += src_regex_SOURCES

###### src/speech #########
src_speech_SOURCES = [
  'src/speech/cst_lpcres.c', 
  'src/speech/cst_track.c', 
  'src/speech/cst_track_io.c', 
  'src/speech/cst_wave.c', 
  'src/speech/cst_wave_io.c', 
  'src/speech/cst_wave_utils.c', 
  'src/speech/g721.c', 
  'src/speech/g723_24.c', 
  'src/speech/g723_40.c', 
  'src/speech/g72x.c', 
  'src/speech/g72x.h', 
  'src/speech/rateconv.c']

libttsmimic_SOURCES += src_speech_SOURCES

###### src/stats #########
src_stats_SOURCES = [ 
  'src/stats/cst_cart.c', 
  'src/stats/cst_ss.c', 
  'src/stats/cst_viterbi.c']

libttsmimic_SOURCES += src_stats_SOURCES

###### src/synth #########
src_synth_SOURCES = [
  'src/synth/cst_ffeatures.c', 
  'src/synth/cst_phoneset.c', 
  'src/synth/cst_ssml.c', 
  'src/synth/cst_synth.c', 
  'src/synth/cst_utt_utils.c', 
  'src/synth/cst_voice.c', 
  'src/synth/mimic.c']
libttsmimic_SOURCES += src_synth_SOURCES

###### src/utils #########
src_utils_SOURCES = [
  'src/utils/cst_alloc.c', 
  'src/utils/cst_args.c', 
  'src/utils/cst_endian.c', 
  'src/utils/cst_error.c', 
  'src/utils/cst_features.c', 
  'src/utils/cst_file_stdio.c', 
  'src/utils/cst_mmap_none.c', 
  'src/utils/cst_mmap_posix.c', 
  'src/utils/cst_mmap_win32.c', 
  'src/utils/cst_plugins.c', 
  'src/utils/cst_socket.c', 
  'src/utils/cst_string.c', 
  'src/utils/cst_tokenstream.c', 
  'src/utils/cst_uregex.c', 
  'src/utils/cst_url.c', 
  'src/utils/cst_val.c', 
  'src/utils/cst_val_const.c', 
  'src/utils/cst_val_user.c', 
  'src/utils/cst_wchar.c']

libttsmimic_SOURCES += src_utils_SOURCES

###### src/wavesynth #########
src_wavesynth_SOURCES = [
  'src/wavesynth/cst_clunits.c', 
  'src/wavesynth/cst_diphone.c', 
  'src/wavesynth/cst_reflpc.c', 
  'src/wavesynth/cst_sigpr.c', 
  'src/wavesynth/cst_sts.c', 
  'src/wavesynth/cst_units.c']



libttsmimic_SOURCES += src_wavesynth_SOURCES


# FIXME: Use DLLs for loading voices
libttsmimic_SOURCES += ['main/mimic_voice_list.c']

# If use plugins, then build the library now, as all languages and voices go to
# shared libraries that depend on this library, otherwise, all languages and voices
# will go to the ttsmimiclib library
if use_plugins

# Math library
m_dep = c_compiler.find_library('m', required : false)

mimic_core_deps += m_dep

libttsmimic_include_dirs += include_directories('include')

ttsmimiclib = library('ttsmimic', libttsmimic_SOURCES,
                      include_directories : libttsmimic_include_dirs,
                      dependencies: mimic_core_deps,
                      install: true)

ttsmimiclib_dep = declare_dependency(link_with : ttsmimiclib, 
                                     include_directories : libttsmimic_include_dirs)

endif

############ Headers #################

ttsmimic_headers = [
  'include/cst_alloc.h',
  'include/cst_args.h',
  'include/cst_audio.h',
  'include/cst_cart.h',
  'include/cst_cg.h',
  'include/cst_clunits.h',
  'include/cst_diphone.h',
  'include/cst_endian.h',
  'include/cst_error.h',
  'include/cst_features.h',
  'include/cst_ffeatures.h',
  'include/cst_file.h',
  'include/cst_hrg.h',
  'include/cst_item.h',
  'include/cst_lexicon.h',
  'include/cst_lts.h',
  'include/cst_lts_rewrites.h',
  'include/cst_phoneset.h',
  'include/cst_plugins.h',
  'include/cst_regex.h',
  'include/cst_relation.h',
  'include/cst_sigpr.h',
  'include/cst_socket.h',
  'include/cst_ss.h',
  'include/cst_string.h',
  'include/cst_sts.h',
  'include/cst_synth.h',
  'include/cst_tokenstream.h',
  'include/cst_track.h',
  'include/cst_units.h',
  'include/cst_uregex.h',
  'include/cst_utterance.h',
  'include/cst_utt_utils.h',
  'include/cst_val_const.h',
  'include/cst_val_defs.h',
  'include/cst_val.h',
  'include/cst_viterbi.h',
  'include/cst_voice.h',
  'include/cst_wave.h',
  'include/cst_wchar.h',
  'include/flite_hts_engine.h',
  'include/mimic.h']

install_headers(ttsmimic_headers, subdir : 'ttsmimic')

########## Man pages ##############

install_man('man/man1/mimic.1')


###### English language support #####################

ttsmimic_cmulex_SOURCES = [
  'lang/cmulex/cmu_lex.c',
  'lang/cmulex/cmu_lex_data.c',
  'lang/cmulex/cmu_lex_entries.c',
  'lang/cmulex/cmu_lts_model.c',
  'lang/cmulex/cmu_lts_model.h',
  'lang/cmulex/cmu_lts_rules.c',
  'lang/cmulex/cmu_postlex.c']

ttsmimic_usenglish_SOURCES = [
  'lang/usenglish/us_aswd.c',
  'lang/usenglish/us_dur_stats.c',
  'lang/usenglish/us_durz_cart.c',
  'lang/usenglish/usenglish.c',
  'lang/usenglish/usenglish_plugin.c',
  'lang/usenglish/us_expand.c',
  'lang/usenglish/us_f0lr.c',
  'lang/usenglish/us_f0_model.c',
  'lang/usenglish/us_ffeatures.c',
  'lang/usenglish/us_gpos.c',
  'lang/usenglish/us_int_accent_cart.c',
  'lang/usenglish/us_int_tone_cart.c',
  'lang/usenglish/us_nums_cart.c',
  'lang/usenglish/us_phoneset.c',
  'lang/usenglish/us_phrasing_cart.c',
  'lang/usenglish/us_pos_cart.c',
  'lang/usenglish/us_text.c']

ttsmimic_usenglish_headers = [
  'lang/usenglish/usenglish.h',
  'lang/usenglish/us_f0.h',
  'lang/usenglish/us_ffeatures.h',
  'lang/usenglish/us_regexes.h',
  'lang/usenglish/us_text.h']
ttsmimic_cmulex_headers = ['lang/cmulex/cmu_lex.h']

# FIXME
#regexes:
#	./make_us_regexes $(BINDIR) >us_regexes.h
#
#us_pos_cart.c: 
#	$(ESTDIR)/../festival/bin/festival \
#           -b $(MIMICDIR)/tools/make_cart.scm \
#               $(MIMICDIR)/tools/make_pos.scm 


install_headers(ttsmimic_usenglish_headers + ttsmimic_cmulex_headers,
                subdir : join_paths('ttsmimic', 'lang', 'en_US'))

ttsmimic_usenglish_includes = libttsmimic_include_dirs + [
  include_directories(join_paths('lang', 'usenglish')),
  include_directories(join_paths('lang', 'cmulex'))
]
conf_data.set('ENABLE_LANG_ENGLISH', 1)

if use_plugins
ttsmimic_usenglish_lib = shared_library('ttsmimic_usenglish',
                                        ttsmimic_usenglish_SOURCES + ttsmimic_cmulex_SOURCES,
                                        include_directories : ttsmimic_usenglish_includes,
                                        dependencies: ttsmimiclib_dep,
                                        install: true)

ttsmimic_usenglish_dep = declare_dependency(link_with : ttsmimic_usenglish_lib, 
                                            include_directories : ttsmimic_usenglish_includes)

else

libttsmimic_include_dirs += ttsmimic_usenglish_includes

libttsmimic_SOURCES += ttsmimic_usenglish_SOURCES + ttsmimic_cmulex_SOURCES

endif

#################### Grapheme support ############

ttsmimic_grapheme_SOURCES = [
  'lang/cmu_grapheme_lang/cmu_grapheme_lang.c',
  'lang/cmu_grapheme_lang/cmu_grapheme_plugin.c',
  'lang/cmu_grapheme_lang/cmu_grapheme_phrasing_cart.c',
  'lang/cmu_grapheme_lang/cmu_grapheme_phoneset.c',
  'lang/cmu_grapheme_lex/cmu_grapheme_lex.c',
  'lang/cmu_grapheme_lex/grapheme_unitran_tables.c'
]

ttsmimic_grapheme_headers = [
  'lang/cmu_grapheme_lang/cmu_grapheme_lang.h',
  'lang/cmu_grapheme_lex/cmu_grapheme_lex.h'
]


install_headers(ttsmimic_grapheme_headers,
                subdir : join_paths('ttsmimic', 'lang', 'grapheme'))

ttsmimic_grapheme_includes = libttsmimic_include_dirs + [
  include_directories(join_paths('lang', 'cmu_grapheme_lang')),
  include_directories(join_paths('lang', 'cmu_grapheme_lex'))
]

conf_data.set('ENABLE_LANG_GRAPHEME', 1)
if use_plugins
ttsmimic_grapheme_lib = shared_library('ttsmimic_grapheme',
                                    ttsmimic_grapheme_SOURCES,
                                    include_directories : ttsmimic_grapheme_includes,
                                    dependencies: ttsmimiclib_dep,
                                    install: true)

ttsmimic_grapheme_dep = declare_dependency(link_with : ttsmimic_grapheme_lib, 
                                        include_directories : ttsmimic_grapheme_includes)

else

libttsmimic_include_dirs += ttsmimic_grapheme_includes

libttsmimic_SOURCES += ttsmimic_grapheme_SOURCES

endif

# mapping:
#	${ESTDIR}/../festival/bin/festival \
#         -b ${TOP}/tools/make_ug.scm \
#            ${FESTVOXDIR}/src/grapheme/unicode_sampa_mapping.scm '(doit)'


############## Indic language support ############

ttsmimic_indic_SOURCES = [
  'lang/cmu_indic_lang/cmu_indic_lang.c',
  'lang/cmu_indic_lang/cmu_indic_plugin.c',
  'lang/cmu_indic_lang/cmu_indic_phoneset.c',
  'lang/cmu_indic_lang/cmu_indic_phrasing_cart.c',
  'lang/cmu_indic_lex/cmu_indic_lex.c'
]

ttsmimic_indic_headers = [
  'lang/cmu_indic_lang/cmu_indic_lang.h',
  'lang/cmu_indic_lex/cmu_indic_lex.h']


install_headers(ttsmimic_indic_headers,
                subdir : join_paths('ttsmimic', 'lang', 'indic'))

ttsmimic_indic_includes = libttsmimic_include_dirs + ttsmimic_usenglish_includes + [
  include_directories(join_paths('lang', 'cmu_indic_lang')),
  include_directories(join_paths('lang', 'cmu_indic_lex'))
]

conf_data.set('ENABLE_LANG_INDIC', 1)
if use_plugins
ttsmimic_indic_lib = shared_library('ttsmimic_indic',
                                    ttsmimic_indic_SOURCES,
                                    include_directories : ttsmimic_indic_includes,
                                    dependencies: [ttsmimic_usenglish_dep, ttsmimiclib_dep],
                                    install: true)

ttsmimic_indic_dep = declare_dependency(link_with : ttsmimic_indic_lib, 
                                        include_directories : ttsmimic_indic_includes)

else

libttsmimic_include_dirs += ttsmimic_indic_includes

libttsmimic_SOURCES += ttsmimic_indic_SOURCES

endif

################# CMU_US_SLT ######################

ttsmimic_voice_cmu_us_slt_SOURCES = [
  'lang/cmu_us_slt/cmu_us_slt.c',
  'lang/cmu_us_slt/cmu_us_slt_cg.c',
  'lang/cmu_us_slt/cmu_us_slt_cg_durmodel.c',
  'lang/cmu_us_slt/cmu_us_slt_cg_f0_trees.c',
  'lang/cmu_us_slt/cmu_us_slt_cg_phonestate.c',
  'lang/cmu_us_slt/cmu_us_slt_cg_single_mcep_trees.c',
  'lang/cmu_us_slt/cmu_us_slt_cg_single_params.c',
  'lang/cmu_us_slt/cmu_us_slt_spamf0_accent.c',
  'lang/cmu_us_slt/cmu_us_slt_spamf0_accent_params.c',
  'lang/cmu_us_slt/cmu_us_slt_spamf0_phrase.c'
]

ttsmimic_voice_cmu_us_slt_includes = [include_directories('lang/cmu_us_slt')]

if use_plugins

ttsmimic_voice_cmu_us_slt_lib = shared_library('ttsmimic_voice_cmu_us_slt',
                                        ttsmimic_voice_cmu_us_slt_SOURCES,
                                        include_directories : ttsmimic_voice_cmu_us_slt_includes,
                                        dependencies: [ttsmimic_usenglish_dep, ttsmimiclib_dep],
                                        install: true)

else


libttsmimic_include_dirs += ttsmimic_voice_cmu_us_slt_includes

libttsmimic_SOURCES += ttsmimic_voice_cmu_us_slt_SOURCES


endif

# Mimic core library, when plugins are disabled

if not use_plugins

# Math library
m_dep = c_compiler.find_library('m', required : false)

mimic_core_deps += m_dep

libttsmimic_include_dirs += include_directories('include')

ttsmimiclib = library('ttsmimic', libttsmimic_SOURCES,
                      include_directories : libttsmimic_include_dirs,
                      dependencies: mimic_core_deps,
                      install: true)

ttsmimiclib_dep = declare_dependency(link_with : ttsmimiclib, 
                                     include_directories : libttsmimic_include_dirs)

endif


########### pkg-config #############

pkg = import('pkgconfig')

pkg.generate(libraries : ttsmimiclib, name : 'mimic',
             description: 'A library for text to speech synthesis.',
             subdirs: 'ttsmimic',
             version: meson.project_version())


############# Binaries ####################

mimicbin = executable('mimic', 'main/mimic_main.c', 
                      include_directories : libttsmimic_include_dirs,
                      dependencies: ttsmimiclib_dep,
                      install: true)
mimicvox_info = executable('mimicvox_info', 'main/mimicvox_info_main.c',
                           include_directories : libttsmimic_include_dirs,
                           dependencies: ttsmimiclib_dep,
                           install: true)

conf_data.set('PACKAGE_NAME', '"' + meson.project_name() + '"')
conf_data.set('PACKAGE_VERSION', '"' + meson.project_version() + '"')
conf_data.set('PACKAGE_URL', '"https://github.com/MycroftAI/mimic/issues"')

configure_file(output : 'config.h',
               configuration : conf_data, install: true, install_dir : 'ttsmimic')


########## Unit tests #########################
unit_tests =  [
  ['hrg', []],
  ['regex', []],
  ['string', []],
  ['token', ['-DTEST_FILE="' + join_paths(meson.current_source_dir(), 'unittests', 'data.one') + '"',
                  '-DTEST_FILE_UTF8="' + join_paths(meson.current_source_dir(), 'unittests', 'data_utf8.txt') + '"']],
  ['voice_select', ['-DVOICE_LIST_DIR="' + join_paths(meson.current_source_dir(), 'voices') + '"',
                    '-DA_VOICE="' + join_paths(meson.current_source_dir(), 'voices', 'cmu_us_rms.flitevox') + '"']],
  ['wave', ['-DVOICE_LIST_DIR="' + join_paths(meson.current_source_dir(), 'voices') + '"',
                 '-DA_VOICE="' + join_paths(meson.current_source_dir(), 'voices', 'cmu_us_rms.flitevox') + '"']],
  ]

foreach test_name_args : unit_tests
   test_bin = executable(test_name_args[0], join_paths('unittests', test_name_args[0] + '_test_main.c'),
                         include_directories : libttsmimic_include_dirs,
                         dependencies: ttsmimiclib_dep,
                         c_args: test_name_args[1])
   test(test_name_args[0], test_bin)
endforeach

########## END Unit tests #########################

########## START testsuite #########################

# These should build, but don't run them
command_line_tools = [
  'asciiS2U', 'asciiU2S', 'bin2ascii', 'combine_waves', 'compare_wave',
  'lpc_resynth', 'lpc_test2', 'lpc_test', 'play_client', 'play_server',
  'play_sync', 'play_wave', 'rfc']


foreach test_name : command_line_tools
   test_bin = executable(test_name, join_paths('testsuite', test_name + '_main.c'),
                         include_directories : libttsmimic_include_dirs,
                         dependencies: [ttsmimiclib_dep, m_dep])
   #test(test_name, test_bin)
endforeach

test_bin = executable('time_voice_load', join_paths('testsuite', 'time_voice_load' + '.c'),
                         include_directories : libttsmimic_include_dirs,
                         dependencies: ttsmimiclib_dep)

########### END testsuite ###########################

