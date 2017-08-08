/*
 * cst_builtin_plugins.c
 * 
 * Copyright 2017 Sergio Oller <sergioller@gmail.com>
 * All rights reserved.
 */
/*
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following disclaimer
 *   in the documentation and/or other materials provided with the
 *   distribution.
 * * Neither the name of the  nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */
 /* This links all the builtin languages and voices, when they are
  * not used as cst_plugins */

#include "config.h"
#include "mimic.h"

#if ENABLE_LANG_ENGLISH_BUILTIN
#include "lang/usenglish/usenglish.h"
#endif

#if ENABLE_LANG_INDIC_BUILTIN
#include "lang/cmu_indic_lang/cmu_indic_lang.h"
#endif

#if ENABLE_LANG_GRAPHEME_BUILTIN
#include "lang/cmu_grapheme_lang/cmu_grapheme_lang.h"
#endif

#if ENABLE_LANG_ES_BUILTIN
#include "lang/es_lang/es_lang.h"
#endif

#if ENABLE_CMU_US_SLT_BUILTIN
#include "lang/cmu_us_slt/cmu_us_slt.h"
#endif

#if ENABLE_CMU_TIME_AWB_BUILTIN
#include "lang/cmu_time_awb/cmu_time_awb.h"
#endif

#if ENABLE_CMU_US_KAL_BUILTIN
#include "lang/cmu_us_kal/cmu_us_kal.h"
#endif

#if ENABLE_CMU_US_KAL16_BUILTIN
#include "lang/cmu_us_kal16/cmu_us_kal16.h"
#endif

#if ENABLE_CMU_US_AWB_BUILTIN
#include "lang/cmu_us_awb/cmu_us_awb.h"
#endif

#if ENABLE_CMU_US_RMS_BUILTIN
#include "lang/cmu_us_rms/cmu_us_rms.h"
#endif

#if ENABLE_CMU_US_SLT_HTS_BUILTIN
#include "lang/cmu_us_slt_hts/cmu_us_slt_hts.h"
#endif

#if ENABLE_VID_GB_AP_BUILTIN
#include "lang/vid_gb_ap/vid_gb_ap.h"
#endif


int mimic_builtin_plugins_init()
{
  #if ENABLE_LANG_ENGLISH_BUILTIN
   usenglish_plugin_init();
  #endif
  #if ENABLE_LANG_INDIC_BUILTIN
   indic_plugin_init();
  #endif
  #if ENABLE_LANG_GRAPHEME_BUILTIN
   grapheme_plugin_init();
  #endif
  #if ENABLE_LANG_ES_BUILTIN
   es_plugin_init();
  #endif
  #if ENABLE_CMU_US_SLT_BUILTIN
   voice_cmu_us_slt_plugin_init();
  #endif
  #if ENABLE_CMU_TIME_AWB_BUILTIN
   voice_cmu_time_awb_plugin_init();
  #endif
  #if ENABLE_CMU_US_KAL_BUILTIN
   voice_cmu_us_kal_plugin_init();
  #endif
  #if ENABLE_CMU_US_KAL16_BUILTIN
   voice_cmu_us_kal16_plugin_init();
  #endif
  #if ENABLE_CMU_US_AWB_BUILTIN
   voice_cmu_us_awb_plugin_init();
  #endif
  #if ENABLE_CMU_US_RMS_BUILTIN
   voice_cmu_us_rms_plugin_init();
  #endif
  #if ENABLE_CMU_US_SLT_HTS_BUILTIN
   voice_cmu_us_slt_hts_plugin_init();
  #endif
  #if ENABLE_VID_GB_AP_BUILTIN
   voice_vid_gb_ap_plugin_init();
  #endif
  return 0;
}

void mimic_builtin_plugins_exit()
{
  #if ENABLE_CMU_US_AWB_BUILTIN
   voice_cmu_us_awb_plugin_exit();
  #endif
  #if ENABLE_CMU_US_RMS_BUILTIN
   voice_cmu_us_rms_plugin_exit();
  #endif
  #if ENABLE_CMU_US_SLT_HTS_BUILTIN
   voice_cmu_us_slt_hts_plugin_exit();
  #endif
  #if ENABLE_VID_GB_AP_BUILTIN
   voice_vid_gb_ap_plugin_exit();
  #endif
  #if ENABLE_CMU_US_KAL16_BUILTIN
   voice_cmu_us_kal16_plugin_exit();
  #endif
  #if ENABLE_CMU_US_KAL_BUILTIN
   voice_cmu_us_kal_plugin_exit();
  #endif
  #if ENABLE_CMU_TIME_AWB_BUILTIN
   voice_cmu_time_awb_plugin_exit();
  #endif
  #if ENABLE_CMU_US_SLT_BUILTIN
   voice_cmu_us_slt_plugin_exit();
  #endif
  #if ENABLE_LANG_ES_BUILTIN
   es_plugin_exit();
  #endif
  #if ENABLE_LANG_GRAPHEME_BUILTIN
   grapheme_plugin_exit();
  #endif
  #if ENABLE_LANG_INDIC_BUILTIN
   indic_plugin_exit();
  #endif
  #if ENABLE_LANG_ENGLISH_BUILTIN
   usenglish_plugin_exit();
  #endif
  return;
}

int mimic_init()
{
    mimic_core_init();
    mimic_builtin_plugins_init();
    mimic_plugins_init();
    return 0;
}

int mimic_exit()
{
    mimic_plugins_exit();
    mimic_builtin_plugins_exit();
    mimic_core_exit();
    return 0;
}

