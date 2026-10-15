// cascade/jtag_cfg.js - the JTAG side of cascade/hardware.py (TI's scripting debugger from
// UniFlash or CCS, run by its node with launcher.mjs; CASCADE_CCS_ROOT is the folder with
// its ccs_base - Scripting.ccs_root()).
//
//   start <firmware.xer5f> <cfg.json>   CPU reset, load the firmware, write gUserCfg from
//                                       cfg.json ({"words": [...]}), run, report status
//   stop <&gUserCfg>                    halt, read the status block and the test log, CPU reset
//                                       (ends streaming)
//   probe                               connect and disconnect, the firmware left as it is
// (&gUserCfg comes from the firmware's symbol table - the session has no symbols here)
//
// Prints one line "RESULT <json>" for the caller, also when the debugger cannot connect.
// "start" never halts the running firmware: a halt makes the R5F miss CSI ping-pong
// interrupts and the chirps of that moment are lost.
const { join, resolve } = require("path");
const fs = require("fs");

const STATUS_FIELDS = ["state", "errStage", "errCode", "framesDone", "heartbeat"];
const CFG_WORDS = 42;                 // magic .. txMaskDev1[11], see cascade_user_cfg.h

const mode = process.argv[3];
const result = { mode };
let ds = null;
let s = null;
const sleep = (ms) => { const until = Date.now() + ms; while (Date.now() < until) { /* wait */ } };

function readStatus(addr) {
    const words = s.memory.read(addr + BigInt(4 * CFG_WORDS), STATUS_FIELDS.length, 32);
    const st = {};
    STATUS_FIELDS.forEach((k, i) => { st[k] = Number(words[i]); });
    if (st.errCode > 0x7FFFFFFF) st.errCode -= 0x100000000;
    return st;
}

function readLog() {
    try {
        const words = s.memory.read(0x88000000n, 0x900, 32);
        let txt = "";
        for (const w of words) for (let k = 0; k < 4; k++) {
            const c = Number((BigInt(w) >> BigInt(8 * k)) & 0xFFn);
            if (c === 0) return txt;
            txt += (c >= 32 && c < 127) || c === 10 ? String.fromCharCode(c) : "";
        }
        return txt;
    } catch (e) { return ""; }
}

try {
    if (!["start", "stop", "probe"].includes(mode)) throw new Error("mode must be start, stop or probe");
    ds = initScripting({ ccsRoot: process.env.CASCADE_CCS_ROOT });
    ds.setScriptingTimeout(60000);
    ds.configure(join(__dirname, "AM273x_XDS110.ccxml"));
    s = ds.openSession(/Cortex_R5_0/);
    // printf goes through CIO (a breakpoint at C$$IO$$) and semihosting breaks at SVC_Handler,
    // which FreeRTOS uses: both would stop the firmware, so neither is armed.
    s.settings.set("AddCIOBreakpointAfterLoad", false);
    s.settings.set("AddSemiHostingBreakpointAfterLoad", false);
    s.target.connect();
    if (mode === "start") {
        const fw = resolve(process.argv[4]);
        const cfg = JSON.parse(fs.readFileSync(resolve(process.argv[5]), "utf8"));
        if (cfg.words.length !== CFG_WORDS) throw new Error(`cfg has ${cfg.words.length} words, expected ${CFG_WORDS}`);
        s.target.halt();
        s.target.reset("CPU Reset");
        s.target.halt();
        s.memory.loadProgram(fw);                      // runs to main
        const addr = BigInt(s.expressions.evaluate("&gUserCfg"));
        // clear the test logger's RAM copy so the log read later is this run's
        s.memory.fill(0x88000000n, 0x900, 0, 32);
        s.memory.write(addr, cfg.words, 32);
        s.memory.write(addr + BigInt(4 * CFG_WORDS), [0, 0, 0, 0, 0], 32);
        const back = s.memory.read(addr, CFG_WORDS, 32).map(Number);
        const bad = back.findIndex((v, i) => v !== (cfg.words[i] >>> 0));
        if (bad >= 0) throw new Error(`gUserCfg word ${bad} reads back ${back[bad]}, wrote ${cfg.words[bad]}`);
        result.gUserCfg = "0x" + addr.toString(16);
        s.target.run(false);
        // no status read here: reads of a running R5F return the debugger's 0xBAD0BAD0
        // pattern, and a halt would cost chirps once streaming starts. The caller watches
        // the DCA1000 data instead and uses "stop" (halted read) to learn what happened.
    } else if (mode === "stop") {
        const addr = BigInt(process.argv[4]);
        s.target.halt();
        result.status = readStatus(addr, false);
        result.log = readLog();
        s.target.reset("CPU Reset");
        s.target.halt();
        result.stopped = true;
    } else {
        result.halted = s.target.isHalted();
    }
    result.ok = true;
} catch (e) {
    result.ok = false;
    result.error = String(e.message || e).split("\n")[0];
}
console.log("RESULT " + JSON.stringify(result));
try { if (s) s.target.disconnect(); } catch (e) { }
try { if (ds) ds.shutdown(); } catch (e) { }
