"""
EEG cognitive battery — response-feedback revision.

Run:
    python eeg_battery.py

Choose Preview or Full in the startup dialog.

Preview:
    Approximately 30-second tasks and 5-second rests.
    No EEG equipment or audio files required.

Full:
    Approximately 5-minute tasks and 2.5-minute rests.
    Requires audio/1.wav through audio/9.wav for auditory PASAT.

PASAT responses remain KEYBOARD input.
Microphone recognition is not implemented.

Hardware triggers remain unconfigured placeholders.
Validate timing on the acquisition computer before EEG use.
"""

from pathlib import Path
from datetime import datetime
import json
import random
import traceback

from psychopy import prefs

prefs.hardware["audioLib"] = ["PTB"]
prefs.hardware["audioLatencyMode"] = 3

from psychopy import core, visual, event, gui
from psychopy.hardware import keyboard


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = (1200, 800)
SCREEN_NUMBER = 0

USE_LSL = False
USE_HARDWARE = False

SPAN_MIN = 5
SPAN_MAX = 7
SPAN_RECALL_TIMEOUT = 30.0

ACK_SECONDS = 0.35

PASAT_INTERVAL = 2.5
PASAT_FEEDBACK_SECONDS = 0.5
AUDIO_SCHEDULE_LEAD = 0.20

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
DATA_DIR = ROOT / "data"

LETTERS = list("BCDFGHJKLMNPQRSTVWXYZ")

WORDS = [
    "BIRD", "CHAIR", "APPLE", "TABLE", "HOUSE", "RIVER",
    "BOOK", "TREE", "SPOON", "HORSE", "SHOE", "TRAIN",
    "BREAD", "CLOCK", "CLOUD", "DOOR", "FISH", "GLASS",
    "HAND", "KEY", "LAMP", "LEAF", "MILK", "MOON",
    "PLATE", "ROAD", "ROCK", "SHIP", "STAR", "STONE",
    "SUN", "TRUCK", "WALL", "WHEEL", "BED", "BOAT",
    "CAKE", "COAT", "CUP", "DESK", "DOG", "FORK",
    "HAT", "LAKE", "PEN", "RING", "SAND", "SOAP",
]

CODES = {
    "task_start": 1,
    "task_end": 2,
    "rest_start": 3,
    "rest_end": 4,
    "trial_start": 5,
    "stimulus_onset": 6,
    "response": 7,
    "stimulus_offset": 8,
    "retention_start": 9,
    "recall_start": 10,
    "submission": 11,
    "experiment_start": 12,
    "experiment_end": 13,
    "abort": 14,
    "task_end_and_rest_start": 15,
}

# Feedback events have names in the local/LSL log but no
# hardware codes yet. Hardware marker allocation and pulse
# collision handling require device-specific validation.


# ============================================================
# HARDWARE TRIGGER PLACEHOLDER
# ============================================================

def hardware_pulse(code):
    """
    Device-specific placeholder: numeric pulse, then reset to zero.

    Do not use core.wait() here. Pulse generation must not block
    screen rendering.

    Confirm the trigger box, protocol, pulse width, and collision
    handling before implementing this function.
    """
    raise NotImplementedError(
        "Hardware trigger adapter is not configured."
    )


# ============================================================
# DESIGN HELPERS
# ============================================================

def make_nback_sequence(n, rng):
    """
    Approximately 30% targets; exactly 45 in a 150-trial block.

    The first two trials cannot be targets.
    Targets never occur on adjacent trials.
    Eligible non-targets sometimes become 1-back or 3-back lures.
    """
    count = round(n * 0.30)

    compressed = sorted(
        rng.sample(range(2, n - count + 1), count)
    )

    targets = {
        value + index
        for index, value in enumerate(compressed)
    }

    sequence = []

    for i in range(n):
        previous = [row["letter"] for row in sequence]

        if i in targets:
            letter = previous[i - 2]
            kind = "target"

        else:
            lures = [
                (previous[i - lag], f"{lag}_back_lure")
                for lag in (1, 3)
                if (
                    i >= lag
                    and (
                        i < 2
                        or previous[i - lag] != previous[i - 2]
                    )
                )
            ]

            if lures and rng.random() < 0.35:
                letter, kind = rng.choice(lures)
            else:
                letter = rng.choice([
                    c for c in LETTERS
                    if c not in previous[-3:]
                ])
                kind = "standard"

        sequence.append({
            "letter": letter,
            "kind": kind,
            "target": kind == "target",
            "match_1_back": (
                i >= 1 and letter == previous[i - 1]
            ),
            "match_3_back": (
                i >= 3 and letter == previous[i - 3]
            ),
        })

    return sequence


def numeric_key(name):
    """Normalize common digit and numeric-keypad key names."""
    if len(name) == 1 and name in "0123456789":
        return name

    for prefix in ("num_", "num", "numpad"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]

            if len(suffix) == 1 and suffix in "0123456789":
                return suffix

    return None


class ExperimentAborted(Exception):
    pass


# ============================================================
# EXPERIMENT ENGINE
# ============================================================

class Battery:
    def __init__(self, participant, seed, preview=True):
        if USE_HARDWARE:
            raise RuntimeError(
                "Hardware output is not configured; "
                "use USE_HARDWARE=False."
            )

        self.preview = preview
        self.task_seconds = 30 if preview else 300
        self.rest_seconds = 5 if preview else 150

        self.rng = random.Random(seed)

        self.clock = core.Clock()
        self.kb = keyboard.Keyboard(clock=self.clock)

        self.events = []
        self.trials = []

        self.task = "setup"
        self.trial = 0

        self.block_start = None
        self.pending_task_start = False

        self.audio = {}
        self.outlet = None

        self.ack_until = -1
        self.ack_visible = False
        self.ack_context = {}

        self.session = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        DATA_DIR.mkdir(exist_ok=True)

        self.output = (
            DATA_DIR / f"battery_{self.session}.json"
        )

        self.meta = {
            "version": "response-feedback-2",
            "participant": participant,
            "seed": seed,
            "preview": preview,
            "task_seconds": self.task_seconds,
            "rest_seconds": self.rest_seconds,
            "codes": CODES,
            "span_min": SPAN_MIN,
            "span_max": SPAN_MAX,
            "ack_seconds": ACK_SECONDS,
            "status": "started",
            "hardware_enabled": USE_HARDWARE,
            "lsl_enabled": USE_LSL,
            "pasat_response_mode": "keyboard",
        }

        if USE_LSL:
            from pylsl import (
                StreamInfo,
                StreamOutlet,
                local_clock,
            )

            self.lsl_clock = local_clock

            info = StreamInfo(
                "CognitiveBatteryMarkers",
                "Markers",
                1,
                0,
                "string",
                f"battery_{self.session}",
            )

            self.outlet = StreamOutlet(info)

        self.win = visual.Window(
            size=WINDOW_SIZE,
            fullscr=not preview,
            screen=SCREEN_NUMBER,
            units="height",
            color=(0, 0, 0),
            colorSpace="rgb",
            waitBlanking=True,
            allowGUI=preview,
        )

        self.mouse = event.Mouse(
            win=self.win,
            visible=False,
        )

        measured = self.win.getActualFrameRate(
            nIdentical=10,
            nMaxFrames=120,
            nWarmUpFrames=20,
        )

        if measured is None and not preview:
            self.win.close()

            raise RuntimeError(
                "No stable refresh rate measured. "
                "Check display settings."
            )

        self.hz = measured or 60.0

        self.meta.update(
            measured_refresh_hz=measured,
            working_refresh_hz=self.hz,
        )

        self.win.refreshThreshold = 1.5 / self.hz
        self.win.recordFrameIntervals = True

        self.fixation = self.text("+", height=0.045)

        self.message = self.text(
            "",
            height=0.033,
        )

        self.feedback = self.text(
            "",
            height=0.040,
            pos=(0, -0.14),
        )

        self.ack = self.text(
            "Response recorded",
            height=0.030,
            pos=(0, -0.18),
        )

        self.letter_stims = {
            c: self.text(c, height=0.13)
            for c in LETTERS
        }

        self.word_stims = {
            w: self.text(w, height=0.065)
            for w in WORDS
        }

        self.digit_stims = {
            d: self.text(str(d), height=0.10)
            for d in range(1, 10)
        }

        self.rotation_stims = {
            (letter, angle, mirror): visual.TextStim(
                self.win,
                text=letter,
                font="Arial",
                height=0.22,
                color="black",
                ori=angle,
                flipHoriz=mirror,
                autoLog=False,
            )
            for letter in ("R", "F", "G")
            for angle in (0, 90, 180, 270)
            for mirror in (False, True)
        }

        self.grid_boxes = []
        self.grid_labels = []

        for row in range(3):
            for col in range(3):
                pos = (
                    (col - 1) * 0.36,
                    (1 - row) * 0.19,
                )

                self.grid_boxes.append(
                    visual.Rect(
                        self.win,
                        width=0.32,
                        height=0.14,
                        pos=pos,
                        lineColor="black",
                        fillColor=(0, 0, 0),
                        autoLog=False,
                    )
                )

                self.grid_labels.append(
                    self.text(
                        "",
                        height=0.035,
                        pos=pos,
                    )
                )

    def text(self, text, height=0.05, pos=(0, 0)):
        return visual.TextStim(
            self.win,
            text=text,
            font="Arial",
            color="black",
            height=height,
            pos=pos,
            wrapWidth=1.3,
            autoLog=False,
        )

    def emit(
        self,
        name,
        aliases=None,
        source_time=None,
        **details,
    ):
        record = {
            "event": name,
            "aliases": aliases or [],
            "code": CODES.get(name),
            "task": self.task,
            "trial": self.trial,
            "dispatch_time": self.clock.getTime(),
            "source_time": source_time,
            **details,
        }

        self.events.append(record)

        if self.outlet is not None:
            self.outlet.push_sample(
                [json.dumps(record)],
                timestamp=self.lsl_clock(),
            )

        if USE_HARDWARE and record["code"] is not None:
            hardware_pulse(record["code"])

    def save(self):
        self.meta["dropped_frames"] = (
            self.win.nDroppedFrames
        )

        payload = {
            "metadata": self.meta,
            "events": self.events,
            "trials": self.trials,
            "frame_intervals_seconds": list(
                self.win.frameIntervals
            ),
        }

        temporary = self.output.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        temporary.replace(self.output)

    def keys(self):
        keys = self.kb.getKeys(
            waitRelease=False,
            clear=True,
        )

        if any(k.name == "escape" for k in keys):
            raise ExperimentAborted

        return keys

    def acknowledge(self):
        self.ack_until = (
            self.clock.getTime() + ACK_SECONDS
        )

        self.ack_context = {
            "response_task": self.task,
            "response_trial": self.trial,
        }

    def draw_ack(self):
        """
        Overlay response feedback without adding trial time.
        Record the feedback's screen-flip boundaries.
        """
        visible = (
            self.clock.getTime() < self.ack_until
        )

        if visible:
            self.ack.draw()

        if visible != self.ack_visible:
            name = (
                "feedback_onset"
                if visible
                else "feedback_offset"
            )

            self.win.callOnFlip(
                self.emit,
                name,
                **self.ack_context,
            )

            self.ack_visible = visible

    def begin_task(self, name):
        self.task = name
        self.trial = 0

        self.block_start = None
        self.pending_task_start = True

        self.ack_until = -1
        self.kb.clearEvents()

    def onset(
        self,
        name,
        aliases=None,
        source_time=None,
        **details,
    ):
        if self.pending_task_start:
            self.block_start = (
                self.clock.getTime()
                if source_time is None
                else source_time
            )

            self.pending_task_start = False

            self.emit(
                "task_start",
                aliases=[name] + (aliases or []),
                source_time=source_time,
                **details,
            )

        else:
            self.emit(
                name,
                aliases=aliases,
                source_time=source_time,
                **details,
            )

    def block_finished(self):
        return (
            self.block_start is not None
            and (
                self.clock.getTime() - self.block_start
                >= self.task_seconds
            )
        )

    def show(
        self,
        draw,
        seconds,
        marker=None,
        aliases=None,
        handler=None,
        clear_keys=False,
        **details,
    ):
        frames = max(
            1,
            round(seconds * self.hz),
        )

        stamp = {}

        def first_flip():
            stamp["time"] = self.clock.getTime()

            if clear_keys:
                self.kb.clearEvents()

            if marker:
                self.onset(
                    marker,
                    aliases=aliases,
                    source_time=stamp["time"],
                    **details,
                )

        for frame in range(frames):
            draw()
            self.draw_ack()

            if frame == 0:
                self.win.callOnFlip(first_flip)

            self.win.flip()

            keys = self.keys()

            if handler and handler(keys, stamp["time"]):
                break

        return stamp["time"]

    def instructions(self, text, rest_ended=False):
        self.ack_until = -1

        self.message.text = (
            text + "\n\nPress SPACE to continue."
        )

        self.message.pos = (0, 0)

        self.mouse.setVisible(False)
        self.kb.clearEvents()

        first = True

        while True:
            self.message.draw()
            self.draw_ack()

            if first and rest_ended:
                self.win.callOnFlip(
                    self.emit,
                    "rest_end",
                )

            self.win.flip()

            if first:
                # Save after the rest-end screen flip,
                # outside timed tasks and rests.
                self.save()

            first = False

            if any(
                k.name == "space"
                for k in self.keys()
            ):
                break

        self.kb.clearEvents()

    def response_event(self, key, onset, **details):
        self.emit(
            "response",
            source_time=key.rt,
            key=key.name,
            rt=key.rt - onset,
            **details,
        )

    def rest(self):
        self.mouse.setVisible(False)
        self.ack_until = -1

        self.show(
            self.fixation.draw,
            self.rest_seconds,
            marker="task_end_and_rest_start",
            aliases=["task_end", "rest_start"],
        )

    def load_audio(self):
        paths = {
            d: AUDIO_DIR / f"{d}.wav"
            for d in range(1, 10)
        }

        missing = [
            str(p)
            for p in paths.values()
            if not p.exists()
        ]

        if missing:
            if not self.preview:
                raise FileNotFoundError(
                    "Full PASAT requires spoken-digit WAV files:\n"
                    + "\n".join(missing)
                )

            self.meta["pasat_modality"] = (
                "visual_preview_substitute"
            )

            return

        from psychopy import sound
        import psychtoolbox as ptb

        self.ptb = ptb

        for digit, path in paths.items():
            snd = sound.Sound(
                str(path),
                preBuffer=-1,
            )

            if snd.getDuration() >= PASAT_INTERVAL:
                raise ValueError(
                    f"{path.name} must be shorter than "
                    "2.5 seconds."
                )

            self.audio[digit] = snd

        self.meta["pasat_modality"] = "auditory"

    # ========================================================
    # TASK 1: VISUAL 2-BACK
    # ========================================================

    def nback(self):
        self.begin_task("visual_2back")

        sequence = make_nback_sequence(
            max(10, round(self.task_seconds / 2)),
            self.rng,
        )

        for index, item in enumerate(sequence):
            self.trial = index + 1

            responses = []
            start = {}

            def collect(keys, phase_onset):
                onset = start.get(
                    "time",
                    phase_onset,
                )

                for key in keys:
                    if (
                        key.name == "space"
                        and key.rt >= onset
                        and not responses
                    ):
                        responses.append(
                            key.rt - onset
                        )

                        self.response_event(
                            key,
                            onset,
                        )

                        self.acknowledge()

                return False

            start["time"] = self.show(
                self.letter_stims[item["letter"]].draw,
                0.5,
                marker="trial_start",
                aliases=["stimulus_onset"],
                handler=collect,
                clear_keys=True,
                **item,
            )

            self.show(
                self.fixation.draw,
                1.5,
                marker="stimulus_offset",
                handler=collect,
            )

            responded = bool(responses)

            if item["target"]:
                outcome = (
                    "hit" if responded else "miss"
                )
            else:
                outcome = (
                    "false_alarm"
                    if responded
                    else "correct_rejection"
                )

            self.trials.append({
                "task": self.task,
                "trial": self.trial,
                **item,
                "onset": start["time"],
                "response_rts": responses,
                "correct": responded == item["target"],
                "outcome": outcome,
            })

    # ========================================================
    # TASK 2: PASAT
    # ========================================================

    def pasat(self):
        self.begin_task("pasat")

        n = max(
            2,
            round(self.task_seconds / PASAT_INTERVAL),
        )

        digits = [
            self.rng.randint(1, 9)
            for _ in range(n)
        ]

        timing_now = (
            self.ptb.GetSecs
            if self.audio
            else self.clock.getTime
        )

        first_time = (
            timing_now() + AUDIO_SCHEDULE_LEAD
        )

        times = [
            first_time + i * PASAT_INTERVAL
            for i in range(n)
        ]

        current = -1
        scheduled = 0

        buffer = ""

        submitted = None
        submitted_rt = None
        onset_exp = None

        feedback_until = -1
        display_text = ""

        self.kb.clearEvents()

        def save_digit():
            if current < 0:
                return

            expected = (
                None
                if current == 0
                else digits[current - 1] + digits[current]
            )

            self.trials.append({
                "task": self.task,
                "trial": current + 1,
                "digit": digits[current],
                "previous_digit": (
                    None
                    if current == 0
                    else digits[current - 1]
                ),
                "expected_sum": expected,
                "response": submitted,
                "unsubmitted_buffer": buffer,
                "rt": submitted_rt,
                "correct": (
                    None
                    if expected is None
                    else submitted == expected
                ),
                "scheduled_onset": onset_exp,
            })

        while (
            timing_now()
            < first_time + n * PASAT_INTERVAL
        ):
            now = timing_now()

            if (
                scheduled < n
                and now >= (
                    times[scheduled] - AUDIO_SCHEDULE_LEAD
                )
            ):
                if self.audio:
                    if now >= times[scheduled]:
                        raise RuntimeError(
                            "Missed audio scheduling deadline; "
                            "check audio setup."
                        )

                    self.audio[digits[scheduled]].play(
                        when=times[scheduled]
                    )

                scheduled += 1

            # These markers correspond to scheduled audio onset.
            # Dispatch lag is recorded; actual acoustic onset
            # must be validated separately.
            if (
                current + 1 < n
                and now >= times[current + 1]
            ):
                save_digit()

                current += 1
                self.trial = current + 1

                buffer = ""
                submitted = None
                submitted_rt = None
                feedback_until = -1

                lag = now - times[current]

                onset_exp = (
                    self.clock.getTime() - lag
                )

                self.onset(
                    "trial_start",
                    aliases=["stimulus_onset"],
                    source_time=onset_exp,
                    digit=digits[current],
                    marker_lateness_seconds=lag,
                    timing_basis=(
                        "scheduled_PTB_audio"
                        if self.audio
                        else "visual_preview"
                    ),
                )

            for key in self.keys():
                if (
                    current < 1
                    or key.rt < onset_exp
                    or key.rt >= (
                        onset_exp + PASAT_INTERVAL
                    )
                    or submitted is not None
                ):
                    continue

                digit = numeric_key(key.name)

                if (
                    digit is not None
                    and len(buffer) < 2
                ):
                    buffer += digit

                    feedback_until = (
                        self.clock.getTime()
                        + PASAT_FEEDBACK_SECONDS
                    )

                    self.response_event(
                        key,
                        onset_exp,
                        buffer=buffer,
                    )

                elif key.name == "backspace":
                    buffer = buffer[:-1]

                    feedback_until = (
                        self.clock.getTime()
                        + PASAT_FEEDBACK_SECONDS
                    )

                    self.response_event(
                        key,
                        onset_exp,
                        buffer=buffer,
                    )

                elif key.name in (
                    "return",
                    "enter",
                    "num_enter",
                ):
                    if (
                        buffer
                        and 2 <= int(buffer) <= 18
                    ):
                        submitted = int(buffer)
                        submitted_rt = (
                            key.rt - onset_exp
                        )

                        self.emit(
                            "submission",
                            source_time=key.rt,
                            answer=submitted,
                            rt=submitted_rt,
                        )

                        feedback_until = (
                            self.clock.getTime()
                            + PASAT_FEEDBACK_SECONDS
                        )

            if (
                not self.audio
                and current >= 0
                and now - times[current] < 0.5
            ):
                self.digit_stims[
                    digits[current]
                ].draw()
            else:
                self.fixation.draw()

            visible = (
                current >= 1
                and self.clock.getTime() < feedback_until
                and now < (
                    times[current]
                    + PASAT_INTERVAL
                    - 0.1
                )
            )

            # Display typed digits without correctness feedback.
            new_text = buffer if visible else ""

            if new_text:
                self.feedback.text = new_text
                self.feedback.draw()

            if new_text != display_text:
                self.win.callOnFlip(
                    self.emit,
                    "pasat_feedback_change",
                    text=new_text,
                )

                display_text = new_text

            self.win.flip()

        save_digit()

        if display_text:
            # The following rest screen removes any
            # remaining response display.
            self.win.callOnFlip(
                self.emit,
                "pasat_feedback_change",
                text="",
            )

    # ========================================================
    # TASK 3: BACKWARD WORD SPAN
    # ========================================================

    def word_span(self):
        self.begin_task("backward_word_span")

        length = SPAN_MIN

        while not self.block_finished():
            self.trial += 1

            pool = self.rng.sample(WORDS, 9)

            targets = pool[:length]
            grid_words = pool[:]

            self.rng.shuffle(grid_words)

            for i, word in enumerate(grid_words):
                self.grid_labels[i].text = word
                self.grid_boxes[i].fillColor = (
                    0, 0, 0
                )

            for position, word in enumerate(targets):
                self.show(
                    self.word_stims[word].draw,
                    1.0,
                    marker=(
                        "trial_start"
                        if position == 0
                        else "stimulus_onset"
                    ),
                    aliases=(
                        ["stimulus_onset"]
                        if position == 0
                        else []
                    ),
                    word=word,
                    serial_position=position + 1,
                    list_length=length,
                )

                if position < length - 1:
                    self.show(
                        lambda: None,
                        0.5,
                        marker="stimulus_offset",
                    )

            self.show(
                self.fixation.draw,
                3.0,
                marker="retention_start",
                aliases=["stimulus_offset"],
            )

            selected = []
            click_rts = []

            previously_down = (
                self.mouse.getPressed()[0]
            )

            self.mouse.setVisible(True)

            self.message.text = (
                f"Select {length} words in REVERSE order"
            )

            self.message.pos = (0, 0.38)

            def draw_grid():
                self.message.draw()

                for box, label in zip(
                    self.grid_boxes,
                    self.grid_labels,
                ):
                    box.draw()
                    label.draw()

            def collect_clicks(keys, onset):
                nonlocal previously_down

                down = self.mouse.getPressed()[0]

                if down and not previously_down:
                    for index, box in enumerate(
                        self.grid_boxes
                    ):
                        word = grid_words[index]

                        if (
                            box.contains(self.mouse)
                            and word not in selected
                        ):
                            selected.append(word)

                            now = self.clock.getTime()
                            click_rts.append(now - onset)

                            box.fillColor = (
                                0.35, 0.35, 0.35
                            )

                            self.grid_labels[index].text = (
                                f"{len(selected)}. {word}"
                            )

                            self.emit(
                                "response",
                                source_time=now,
                                response_type="mouse_click",
                                word=word,
                                selection_position=len(selected),
                                rt=now - onset,
                            )

                            break

                previously_down = down

                return len(selected) == length

            self.show(
                draw_grid,
                SPAN_RECALL_TIMEOUT,
                marker="recall_start",
                handler=collect_clicks,
                clear_keys=True,
            )

            self.mouse.setVisible(False)
            self.message.pos = (0, 0)

            expected = list(reversed(targets))
            correct = selected == expected

            self.trials.append({
                "task": self.task,
                "trial": self.trial,
                "list_length": length,
                "targets": targets,
                "grid_words": grid_words,
                "expected": expected,
                "selected": selected,
                "click_rts": click_rts,
                "correct": correct,
                "timed_out": len(selected) < length,
            })

            if correct:
                length = min(
                    SPAN_MAX,
                    length + 1,
                )

            self.show(
                self.fixation.draw,
                0.5,
            )

    # ========================================================
    # TASK 4: MENTAL ROTATION
    # ========================================================

    def rotation(self):
        self.begin_task("mental_rotation")

        # Finish the current 8-trial set to preserve the
        # normal/mirrored and angle balance.
        while not self.block_finished():
            conditions = [
                (angle, mirrored)
                for angle in (0, 90, 180, 270)
                for mirrored in (False, True)
            ]

            self.rng.shuffle(conditions)

            for angle, mirrored in conditions:
                self.trial += 1

                letter = self.rng.choice(
                    ("R", "F", "G")
                )

                answer = {}

                def collect(keys, onset):
                    for key in keys:
                        if (
                            key.name in ("left", "right")
                            and key.rt >= onset
                            and not answer
                        ):
                            answer.update(
                                key=key.name,
                                rt=key.rt - onset,
                            )

                            self.response_event(
                                key,
                                onset,
                            )

                            self.acknowledge()

                            return True

                    return False

                onset = self.show(
                    self.rotation_stims[
                        (letter, angle, mirrored)
                    ].draw,
                    3.0,
                    marker="trial_start",
                    aliases=["stimulus_onset"],
                    handler=collect,
                    clear_keys=True,
                    letter=letter,
                    angle=angle,
                    mirrored=mirrored,
                )

                expected = (
                    "right" if mirrored else "left"
                )

                self.trials.append({
                    "task": self.task,
                    "trial": self.trial,
                    "letter": letter,
                    "angle": angle,
                    "mirrored": mirrored,
                    "expected_key": expected,
                    "response": answer.get("key"),
                    "rt": answer.get("rt"),
                    "correct": (
                        answer.get("key") == expected
                    ),
                    "onset": onset,
                })

                # Response acknowledgment appears during this
                # existing ITI, adding no extra time.
                self.show(
                    self.fixation.draw,
                    1.0,
                    marker="stimulus_offset",
                )

    # ========================================================
    # COMPLETE EXPERIMENT
    # ========================================================

    def run(self):
        try:
            self.load_audio()

            self.emit("experiment_start")

            mode = (
                "PREVIEW: approximately 30 seconds per task; "
                "5-second rests."
                if self.preview
                else
                "FULL RUN: approximately 5 minutes per task; "
                "2.5-minute rests."
            )

            self.instructions(
                "COGNITIVE TASK BATTERY\n\n"
                + mode
                + "\n\nDuring rest, keep your eyes open, "
                "look at the cross, and relax."
                "\nEEG recording should continue through "
                "the tasks and rests."
                "\n\nPress ESCAPE at any time to stop."
            )

            self.instructions(
                "VISUAL 2-BACK\n\n"
                "Press SPACE only when the letter matches "
                "the letter TWO positions earlier."
                "\n\nExample: B - K - B"
                "\nPress SPACE for the second B."
                "\n\nDo not respond to the first two letters. "
                "You may respond during the letter or the "
                "following cross."
                '\n\n"Response recorded" confirms your first '
                "Space press was saved; it does not "
                "indicate correctness."
            )

            self.nback()
            self.rest()

            substitute = (
                "\n\nPREVIEW ONLY: digits appear visually "
                "because spoken-digit recordings were not found."
                if not self.audio
                else ""
            )

            self.instructions(
                "SERIAL ADDITION\n\n"
                "Add the current digit to the immediately "
                "previous digit. Do not keep a running total."
                "\n\nExample: 3, 5, 2 -> answers 8, then 7."
                "\n\nTYPE the sum and press ENTER before "
                "the next digit."
                "\nYour typed digits briefly appear to "
                "confirm the keys registered."
                "\nUse BACKSPACE to correct an entry "
                "before submission."
                "\nDo not respond to the first digit."
                + substitute,
                rest_ended=True,
            )

            self.pasat()
            self.rest()

            self.instructions(
                "BACKWARD WORD SPAN\n\n"
                "Remember the words as they appear and "
                "during the fixation cross."
                "\n\nClick the words in REVERSE order on the grid."
                "\nExample: BIRD, CHAIR, APPLE -> "
                "click APPLE, CHAIR, BIRD."
                "\n\nEach word can be selected once. "
                "Selections cannot be undone."
                "\nLists start at FIVE words and grow after "
                "a completely correct response, up to SEVEN."
                "\nYou have 30 seconds to select the list.",
                rest_ended=True,
            )

            self.word_span()
            self.rest()

            self.instructions(
                "MENTAL ROTATION\n\n"
                "Mentally rotate the character back to upright."
                "\n\nLEFT arrow = normal character."
                "\nRIGHT arrow = mirror-reversed character."
                "\n\nRespond accurately and quickly. "
                "You have up to three seconds."
                '\n"Response recorded" confirms the key was '
                "saved; it does not indicate correctness.",
                rest_ended=True,
            )

            self.rotation()

            self.ack_until = -1

            self.message.text = (
                "Experiment complete."
                "\n\nThank you."
                "\n\nPress SPACE to close."
            )

            self.message.draw()
            self.draw_ack()

            self.win.callOnFlip(
                self.emit,
                "task_end",
            )

            self.win.flip()

            self.meta["status"] = "completed"

            self.emit("experiment_end")
            self.save()
            self.kb.clearEvents()

            while True:
                self.message.draw()
                self.win.flip()

                if any(
                    k.name == "space"
                    for k in self.keys()
                ):
                    break

        except ExperimentAborted:
            if self.meta["status"] != "completed":
                self.meta["status"] = "aborted"
                self.emit("abort")

        except Exception:
            self.meta["status"] = "error"
            self.meta["error"] = traceback.format_exc()
            raise

        finally:
            try:
                for snd in self.audio.values():
                    snd.stop()

                self.save()

            finally:
                self.win.close()

            print(
                f"\nData saved to:\n{self.output}\n"
            )


# ============================================================
# STARTUP DIALOG
# ============================================================

if __name__ == "__main__":
    info = {
        "participant": "preview",
        "seed": random.randrange(1, 2**31),
        "run_mode": [
            "Preview (30-second tasks)",
            "Full (5-minute tasks)",
        ],
    }

    dialog = gui.DlgFromDict(
        info,
        title="EEG Cognitive Battery",
        order=[
            "participant",
            "seed",
            "run_mode",
        ],
    )

    if dialog.OK:
        experiment = Battery(
            participant=str(info["participant"]),
            seed=int(info["seed"]),
            preview=str(
                info["run_mode"]
            ).startswith("Preview"),
        )

        experiment.run()

    core.quit()