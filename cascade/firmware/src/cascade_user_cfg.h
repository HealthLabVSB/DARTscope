/*
 * cascade_user_cfg.h - radar parameters written by the host over JTAG before the firmware
 * runs (cascade_gui.py / jtag_cfg_run.js), plus the status the firmware reports back.
 *
 * The layout is mirrored in cascade_core.py (USER_CFG_FIELDS): every field is a uint32_t,
 * in this order. Change both together and bump USER_CFG_VERSION.
 */
#ifndef CASCADE_USER_CFG_H
#define CASCADE_USER_CFG_H

#include <stdint.h>

#define USER_CFG_MAGIC          (0xCA5CAD01U)
#define USER_CFG_VERSION        (2U)
#define USER_CFG_MAX_SAMPLES    (512U)      /* CSI ping/pong buffers are sized for this */
#define USER_CFG_MAX_CHIRPS     (12U)       /* chirp configs per device (8 KB mmWave heap) */

/* status.state */
#define USER_STATE_LOADED       (0U)
#define USER_STATE_CONFIGURING  (1U)
#define USER_STATE_RUNNING      (2U)
#define USER_STATE_DONE         (3U)
#define USER_STATE_ERROR        (4U)

/* status.errStage */
#define USER_ERR_NONE           (0U)
#define USER_ERR_CFG_INVALID    (1U)
#define USER_ERR_INIT           (2U)
#define USER_ERR_OPEN           (3U)
#define USER_ERR_PROFILE        (4U)
#define USER_ERR_CHIRP          (5U)
#define USER_ERR_CONFIG         (6U)
#define USER_ERR_START          (7U)
#define USER_ERR_STOP           (8U)
#define USER_ERR_LVDS           (9U)

typedef struct CascadeUserCfg_t
{
    /* written by the host */
    uint32_t magic;             /* USER_CFG_MAGIC once the host has written the block   */
    uint32_t version;           /* USER_CFG_VERSION                                     */
    uint32_t startFreqConst;    /* 3.6e9 / 2^26 Hz per LSB                              */
    uint32_t freqSlopeConst;    /* int16, 3.6e9 * 900 / 2^26 Hz/us per LSB              */
    uint32_t idleTime;          /* 10 ns                                                */
    uint32_t adcStartTime;      /* 10 ns                                                */
    uint32_t rampEndTime;       /* 10 ns                                                */
    uint32_t txStartTime;       /* int16, 10 ns                                         */
    uint32_t numAdcSamples;     /* complex samples per chirp per RX                     */
    uint32_t digOutSampleRate;  /* ksps                                                 */
    uint32_t rxGain;            /* dB (rlProfileCfg_t.rxGain)                           */
    uint32_t hpfCornerFreq1;    /* RL_RX_HPF1_*                                         */
    uint32_t hpfCornerFreq2;    /* RL_RX_HPF2_*                                         */
    uint32_t txOutPowerBackoffCode;
    uint32_t numChirpsInLoop;   /* 1..USER_CFG_MAX_CHIRPS                               */
    uint32_t numLoops;          /* 1..255                                               */
    uint32_t framePeriodicity;  /* 5 ns                                                 */
    uint32_t numFrames;         /* frames to record (1..65535)                          */
    uint32_t txMaskDev0[USER_CFG_MAX_CHIRPS];   /* TX enable per chirp, primary AWR2243   */
    uint32_t txMaskDev1[USER_CFG_MAX_CHIRPS];   /* TX enable per chirp, secondary AWR2243 */

    /* written by the firmware */
    uint32_t state;             /* USER_STATE_*                                         */
    uint32_t errStage;          /* USER_ERR_*                                           */
    int32_t  errCode;           /* mmWave error code of the failing call                */
    uint32_t framesDone;        /* CSI end-of-frame count                               */
    uint32_t heartbeat;         /* increments while the test task waits for frames      */
} CascadeUserCfg;

extern volatile CascadeUserCfg gUserCfg;

#endif
