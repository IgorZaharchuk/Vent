#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.0.7 - ФИНАЛЬНАЯ ВЕРСИЯ
- обучение k_heat с отбраковкой выбросов
- компактный лог без дублей
- исправлен дубль прогноза
"""
import json
import time
import math
import os
import threading
import requests
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from collections import deque

# ==================== КОНСТАНТЫ ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(SCRIPT_DIR, 'vent_knowledge.json')
STATE_PATH = os.path.join(SCRIPT_DIR, 'vent_state.json')
DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, 'vent_debug.log')

PARAMS = {
    'k_heat_init': 0.000107,
    'min_bed': 50,
    'max_bed': 120,
    'adapt_start': 60.0,
    'error5_start': 90.0,
    'error5_margin': 3.0,
    'cooldown_sec': 45.0,
    'adapt_step': 15,
    'bed_stable': 3.0,
    'max_sessions': 30,
    'history_points': 100,
    'rate_history_size': 5,
}

# ==================== ЛОГГЕР ====================
class Logger:
    def __init__(self, max_bytes=524288):
        self.logger = logging.getLogger('vent_calc')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        formatter.converter = lambda *args: time.gmtime(time.time() + 3*3600)
        
        handler = RotatingFileHandler(
            DEBUG_LOG_PATH, mode='a', maxBytes=max_bytes,
            backupCount=0, encoding='utf-8'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def startup(self, k_heat, samples):
        self.logger.info(f"vent_calc v1.0.7 | k_heat={k_heat:.6f} 1/сек | семпл={samples}")
    
    def status(self, sessions, samples):
        self.logger.info(f"📚 {sessions} сес | семпл={samples}")
    
    def init(self, bed, k_heat, amb, pred_time):
        self.logger.info(f"🔄 init bed={bed:.0f}°C | k_heat={k_heat:.6f} | amb={amb:.1f}°C | прогноз={pred_time:.0f}мин")
    
    # ИСПРАВЛЕНО: убрали pred=, добавили valid_str
    def predict_full(self, pred, valid, elapsed_sec, time_left, target_time, method):
        valid_str = "✓" if valid else "✗"
        self.logger.info(
            f"📊 t={elapsed_sec:4.0f}с | left={int(time_left)} | target={int(target_time)} | "
            f"{pred:3}мин [{valid_str}] ({method})"
        )
    
    def learned(self, param, old, new, source):
        self.logger.info(f"📈 {param}: {old:.6f}→{new:.6f} ({source})")
    
    def adapt_up(self, old, new, pred, target_time):
        self.logger.info(f"⬆️ {old:.0f}→{new:.0f}°C | pred={pred:.0f}мин > {target_time:.0f}мин")
    
    def adapt_down(self, old, new, pred, target_time):
        self.logger.info(f"⬇️ {old:.0f}→{new:.0f}°C | pred={pred:.0f}мин < {target_time:.0f}мин")
    
    def adapt_force(self, old, new, reason):
        self.logger.info(f"🔥 {old:.0f}→{new:.0f}°C | {reason}")
    
    # ИСПРАВЛЕНО: убрали pred и valid
    def result(self, temp, rate, bed_set, bed_real, stable):
        rate_str = f"{rate:.2f}" if abs(rate) > 0.01 else "0.00"
        stable_str = "стаб" if stable else "нагр"
        
        self.logger.info(
            f"✅ {temp:5.1f}°C | {rate_str:5}°C/мин | "
            f"ст={bed_set:3.0f}/{bed_real:3.0f}°C | {stable_str:4}"
        )

log = Logger()

# ==================== ФИЛЬТР КАЛМАНА ====================
class KalmanFilter:
    def __init__(self, init_temp, dt=10.0):
        self.x = np.array([init_temp, 0.0])
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.P = np.eye(2) * 100.0
        self.R = 0.1
        self.Q = np.diag([0.001, 0.0001])
        self.rate_ema = 0.0
        
    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H.flatten())) @ self.P
        
        raw_rate = self.x[1] * 60.0
        self.rate_ema = 0.1 * raw_rate + 0.9 * self.rate_ema
        
        return self.x[0], self.rate_ema

# ==================== ФИЗИЧЕСКАЯ МОДЕЛЬ ====================
class PhysicsModel:
    def __init__(self, k_heat=PARAMS['k_heat_init']):
        self.k_heat = k_heat
    
    def time_to_target(self, bed, current, target):
        if target >= bed:
            return None
        try:
            ratio = (bed - target) / (bed - current)
            if ratio <= 0 or ratio >= 1:
                return None
            t_sec = -math.log(ratio) / self.k_heat
            return t_sec / 60.0
        except Exception:
            return None

    def bed_needed(self, current, target, target_minutes):
        t_sec = target_minutes * 60
        exp_kt = math.exp(-self.k_heat * t_sec)
        if exp_kt >= 0.99:
            return PARAMS['max_bed']
        bed = (target - current * exp_kt) / (1 - exp_kt)
        return max(PARAMS['min_bed'], min(PARAMS['max_bed'], bed))

    def update_k_heat(self, bed, T0, T1, dt):
        if abs(bed - T0) < 1 or bed <= T1:
            return None
        try:
            ratio = (bed - T1) / (bed - T0)
            if 0.01 < ratio < 0.99:
                new_k = -math.log(ratio) / dt
                return new_k
        except Exception:
            pass
        return None

    def get_k_eq(self):
        return 0.33 + 1800 * self.k_heat

    def is_reachable(self, amb, target):
        k_eq = self.get_k_eq()
        k_eq = max(0.3, min(0.6, k_eq))
        max_eq = amb + k_eq * (PARAMS['max_bed'] - amb)
        return target <= max_eq + 2.0

# ==================== БАЗА ЗНАНИЙ ====================
class KnowledgeBase:
    def __init__(self):
        self.data = self._load()
        self.physics = PhysicsModel(self.get_k_heat())
        self.last_logged_sessions = len(self.data.get('sessions', []))
    
    def _load(self):
        if os.path.exists(KNOWLEDGE_PATH):
            try:
                with open(KNOWLEDGE_PATH) as f:
                    data = json.load(f)
                    if data.get('version') == '2.0' and 'sessions' in data:
                        return data
            except Exception as e:
                log.logger.error(f"Ошибка загрузки: {e}")
                backup = KNOWLEDGE_PATH + '.bak.' + str(int(time.time()))
                try:
                    os.rename(KNOWLEDGE_PATH, backup)
                    log.logger.info(f"📦 Создан бэкап: {backup}")
                except:
                    pass
        return self._create_default()
    
    def _create_default(self):
        return {
            "version": "2.0",
            "k_heat": PARAMS['k_heat_init'],
            "samples": [],
            "sessions": []
        }
    
    def _save(self):
        with lock:
            with open(KNOWLEDGE_PATH + '.tmp', 'w') as f:
                json.dump(self.data, f, indent=2)
            os.replace(KNOWLEDGE_PATH + '.tmp', KNOWLEDGE_PATH)
    
    def get_k_heat(self):
        return self.data.get('k_heat', PARAMS['k_heat_init'])
    
    def get_samples_count(self):
        return len(self.data.get('samples', []))
    
    def _cleanup_sessions(self):
        if len(self.data['sessions']) <= PARAMS['max_sessions']:
            return
        sessions_with_points = [(s, len(s.get('history', []))) for s in self.data['sessions']]
        sessions_with_points.sort(key=lambda x: x[1])
        to_remove = len(self.data['sessions']) - PARAMS['max_sessions']
        remove_ids = [s[0]['id'] for s in sessions_with_points[:to_remove]]
        self.data['sessions'] = [s for s in self.data['sessions'] if s['id'] not in remove_ids]
        log.logger.info(f"🧹 Удалено {to_remove} пустых сессий")
    
    def start_session(self, target, bed, amb):
        sid = len(self.data['sessions']) + 1
        self.data['sessions'].append({
            'id': sid,
            'start': time.time(),
            'target': target,
            'bed_target': bed,
            'ambient': amb,
            'success': False,
            'history': []
        })
        self._cleanup_sessions()
        self._save()
        return sid
    
    def add_point(self, sid, elapsed, chamber, bed):
        for s in self.data['sessions']:
            if s['id'] == sid:
                s['history'].append({
                    't': round(elapsed, 1),
                    'chamber': round(chamber, 2),
                    'bed': round(bed, 2)
                })
                self._save()
                return
    
    def finish(self, sid, success, ttt=None):
        for s in self.data['sessions']:
            if s['id'] == sid:
                s['end'] = time.time()
                s['success'] = success
                if ttt:
                    s['time_to_target'] = round(ttt, 1)
                self._learn_from_session(s)
                self._save()
                return
    
    def _learn_from_session(self, session):
        history = session['history']
        if len(history) < 5:
            return
        
        k_values = []
        bed_target = session['bed_target']

        for i in range(len(history) - 3):
            p1 = history[i]
            p2 = history[i + 2]
            dt = p2['t'] - p1['t']
            
            if dt < 20 or dt > 60:
                continue
            
            if abs(p2['bed'] - p1['bed']) > 5:
                continue
            
            if abs(p1['bed'] - bed_target) > 5 or abs(p2['bed'] - bed_target) > 5:
                continue
            
            bed = (p1['bed'] + p2['bed']) / 2
            if bed <= p1['chamber'] or bed <= p2['chamber']:
                continue

            ratio = (bed - p2['chamber']) / (bed - p1['chamber'])
            if not (0.01 < ratio < 0.99):
                continue

            new_k = -math.log(ratio) / dt
            if 0.00008 < new_k < 0.00035:
                k_values.append(new_k)

        if k_values:
            median_k = np.median(k_values)
            self.data['samples'].append(median_k)
            if len(self.data['samples']) > 500:
                self.data['samples'] = self.data['samples'][-500:]
            self._recalibrate()
    
    def _recalibrate(self):
        if len(self.data['samples']) >= 15:
            new_k = np.median(self.data['samples'][-100:])
            old_k = self.data['k_heat']
            self.data['k_heat'] = old_k * 0.9 + new_k * 0.1
            self.physics.k_heat = self.data['k_heat']
            self._save()
            log.logger.info(f"📊 Калибровка k_heat: {old_k:.6f}→{self.data['k_heat']:.6f}")
    
    def get_history(self, sid, max_points=10):
        for s in self.data['sessions']:
            if s['id'] == sid:
                h = s['history'][-max_points:]
                return [p['t'] for p in h], [p['chamber'] for p in h]
        return [], []
    
    def maybe_log_sessions(self):
        current = len(self.data.get('sessions', []))
        if current != self.last_logged_sessions:
            log.logger.info(f"✅ Загружена база знаний v2.0, сессий: {current}")
            self.last_logged_sessions = current

# ==================== MOONRAKER ====================
def moonraker_get(path):
    try:
        return requests.get(f"http://127.0.0.1:7125{path}", timeout=3).json()
    except:
        return None

def send_vars(vars_dict):
    if not vars_dict:
        return
    script = "\n".join([
        f'SET_GCODE_VARIABLE MACRO="_VENT_VAR" VARIABLE="{k}" VALUE={v}'
        for k, v in vars_dict.items()
    ])
    try:
        requests.post("http://127.0.0.1:7125/printer/gcode/script", 
                     json={"script": script}, timeout=3)
    except:
        pass

# ==================== СОСТОЯНИЕ ====================
@dataclass
class State:
    session_id: Optional[int] = None
    call_count: int = 0
    kalman: Optional[Dict] = None
    last_pred: int = 0
    last_rate: float = 0.0
    cooldown_start: float = 0
    last_adapt: float = 0
    error: int = 0
    rate_history: deque = field(default_factory=lambda: deque(maxlen=PARAMS['rate_history_size']))
    startup_logged: bool = False

    def add_rate(self, rate):
        self.rate_history.append(rate)
    
    def all_rates_low(self, threshold=0.05):
        if not self.rate_history:
            return False
        return all(r < threshold for r in self.rate_history)
    
    def update_cooldown(self, rate, now):
        if rate < 0.01:
            if self.cooldown_start == 0:
                self.cooldown_start = now
        else:
            self.cooldown_start = 0
    
    def can_adapt(self, now, min_interval=180):
        return now - self.last_adapt > min_interval

def load_state():
    state = State()
    if not os.path.exists(STATE_PATH):
        return state
    try:
        with open(STATE_PATH, 'r') as f:
            content = f.read().strip()
            if not content:
                return state
            data = json.loads(content)
            for k, v in data.items():
                if hasattr(state, k):
                    if k == 'rate_history' and v:
                        state.rate_history = deque(v, maxlen=PARAMS['rate_history_size'])
                    else:
                        setattr(state, k, v)
            return state
    except Exception:
        return state

def save_state(state):
    try:
        state_dict = asdict(state)
        state_dict['rate_history'] = list(state.rate_history)
        with open(STATE_PATH + '.tmp', 'w') as f:
            json.dump(state_dict, f)
        os.replace(STATE_PATH + '.tmp', STATE_PATH)
    except Exception:
        pass

# ==================== ОСНОВНАЯ ЛОГИКА ====================
def process_heating(input_vars):
    start = time.time()
    kb = KnowledgeBase()
    state = load_state()
    
    control = int(input_vars.get("control_mode", 0))
    target_reached = int(input_vars.get("target_reached", 0))
    wait_start = int(input_vars.get("wait_start_loop", -1))
    last_time = float(input_vars.get("last_time_for_rate", 0))
    loop = int(input_vars.get("loop_counter", 0))
    
    elapsed = (loop - wait_start) * 10.0 if wait_start >= 0 else 0
    
    if last_time == -1.0 and control != 3:
        state = State()
        save_state(state)
        return {"last_time_for_rate": 1.0, "wait_start_loop": wait_start}
    
    if target_reached == 1:
        if state.session_id:
            kb.finish(state.session_id, True, elapsed)
        state = State()
        save_state(state)
        return {"last_time_for_rate": loop, "wait_start_loop": -1}
    
    if wait_start == -1 and state.session_id:
        log.logger.info(f"🔌 VENT_OFF - сес #{state.session_id} завершена")
        kb.finish(state.session_id, False)
        state = State()
        save_state(state)
        return {"last_time_for_rate": 1.0, "wait_start_loop": -1}
    
    chamber_target = float(input_vars.get("chamber_target", 40))
    bed_target = float(input_vars.get("bed_target_temp", 50))
    ambient = float(input_vars.get("start_chamber_temp", 25))
    bed_min = float(input_vars.get("bed_min_temp", 50))
    real_bed = float(input_vars.get("real_bed_temp", 0))
    current = float(input_vars.get("ema_chamber_temp", 25))
    median = float(input_vars.get("current_median_temp", current))
    time_left = float(input_vars.get("time_left_sec", 0)) / 60
    
    if not state.kalman:
        if not kb.physics.is_reachable(ambient, chamber_target):
            k_eq = kb.physics.get_k_eq()
            max_eq = ambient + k_eq * (PARAMS['max_bed'] - ambient)
            log.logger.warning(f"⚠️ цель {chamber_target}°C недостиж (max {max_eq:.1f}°C)")
            return {"error_code": 6, "equilibrium_est": max_eq,
                    "wait_start_loop": loop, "last_time_for_rate": loop}
        
        if time_left > 0:
            target_time = time_left / 2
            bed_needed = kb.physics.bed_needed(median, chamber_target, target_time)
            bed_init = min(max(bed_needed, bed_min, real_bed), PARAMS['max_bed'])
            init_pred = kb.physics.time_to_target(bed_init, median, chamber_target)
        else:
            bed_init = max(bed_min, real_bed)
            init_pred = None
        
        state.session_id = kb.start_session(chamber_target, bed_init, ambient)
        state.kalman = {"x": [median, 0.0], "P": [[100, 0], [0, 100]]}
        
        if not state.startup_logged:
            log.startup(kb.physics.k_heat, kb.get_samples_count())
            log.status(len(kb.data.get('sessions', [])), kb.get_samples_count())
            state.startup_logged = True
        
        save_state(state)
        
        log.init(bed_init, kb.physics.k_heat, ambient, init_pred or 0)
        return {"bed_target_temp": round(bed_init, 0),
                "wait_start_loop": loop,
                "last_time_for_rate": loop}
    
    kf = KalmanFilter(0)
    kf.x = np.array(state.kalman['x'])
    kf.P = np.array(state.kalman['P'])
    filtered, display_rate = kf.update(median)
    state.kalman = {"x": kf.x.tolist(), "P": kf.P.tolist()}
    
    state.add_rate(display_rate)
    state.last_rate = display_rate
    state.call_count += 1
    
    if state.session_id:
        kb.add_point(state.session_id, elapsed, median, real_bed)
    
    k_heat = kb.physics.k_heat
    bed_stable = real_bed >= bed_target - PARAMS['bed_stable']
    
    state.update_cooldown(display_rate, elapsed)
    cooldown_time = state.cooldown_start and (elapsed - state.cooldown_start)
    
    pred = None
    method = None
    pred = kb.physics.time_to_target(bed_target, filtered, chamber_target)
    if pred:
        method = "physical"
    
    valid = 1 if pred and 5 < pred < 180 else 0
    if valid:
        pred_int = int(round(pred))
        state.last_pred = pred_int
    else:
        pred_int = state.last_pred
    
    error = int(input_vars.get("error_code", 0))
    
    result = {
        "heating_estimate_valid": valid,
        "last_target_minutes": pred_int,
        "current_heat_rate_cpm": round(display_rate, 2),
        "bed_target_temp": bed_target,
        "wait_start_loop": wait_start,
        "last_time_for_rate": loop,
        "error_code": error
    }
    
    # ИСПРАВЛЕНО: вызываем новый метод predict_full с valid
    if valid:
        target_time = max(time_left / 2.0, 5.0)
        log.predict_full(pred_int, valid, elapsed, time_left, target_time, method)
    
    new_bed = bed_target
    adapted = False
    
    if bed_stable and valid and elapsed >= PARAMS['adapt_start'] and state.can_adapt(elapsed):
        target_time = time_left / 2
        if pred_int > target_time * 1.2:
            bed_needed = kb.physics.bed_needed(filtered, chamber_target, target_time)
            if bed_needed > bed_target + 5:
                new_bed = min(round(bed_needed / 5) * 5, PARAMS['max_bed'])
                new_bed = max(new_bed, bed_min)
                adapted = True
                log.adapt_up(bed_target, new_bed, pred_int, target_time)
        elif pred_int < target_time * 0.8 and bed_target > bed_min:
            new_bed = max(bed_target - 5, bed_min)
            adapted = True
            log.adapt_down(bed_target, new_bed, pred_int, target_time)
        if adapted:
            state.last_adapt = elapsed
    
    if not adapted and cooldown_time and cooldown_time >= PARAMS['cooldown_sec']:
        if state.all_rates_low(0.05):
            if state.can_adapt(elapsed, 120):
                new_bed = min(bed_target + PARAMS['adapt_step'], PARAMS['max_bed'])
                new_bed = max(new_bed, bed_min)
                adapted = True
                state.last_adapt = elapsed
                log.adapt_force(bed_target, new_bed, f"cooldown {cooldown_time:.0f}с")
    
    if adapted:
        result["bed_target_temp"] = round(new_bed, 0)
    
    if bed_stable and valid and elapsed >= PARAMS['error5_start'] and pred_int:
        if (pred_int - PARAMS['error5_margin']) > time_left:
            if new_bed >= 110.0 and error != 5:
                result["error_code"] = 5
                log.logger.warning(f"⚠️ e5: нужно {pred_int}мин, ост {time_left:.0f}мин")
        elif error == 5 and pred_int <= time_left * 0.8:
            result["error_code"] = 0
            log.logger.info(f"✅ e5 cleared")
    
    samples = kb.get_samples_count()
    # ИСПРАВЛЕНО: вызываем новый метод result без pred и valid
    log.result(filtered, display_rate, result['bed_target_temp'], real_bed, bed_stable)
    
    save_state(state)
    return result

# ==================== MAIN ====================
lock = threading.RLock()

def main():
    kb = KnowledgeBase()
    kb.maybe_log_sessions()
    
    obj = moonraker_get("/printer/objects/query?gcode_macro%20_VENT_VAR")
    if not obj:
        log.logger.error("moonraker unreachable")
        return
    
    macro = obj.get('result', {}).get('status', {}).get('gcode_macro _VENT_VAR', {})
    if not macro:
        log.logger.error("_VENT_VAR not found")
        return
    
    keys = ['control_mode', 'target_reached', 'wait_start_loop', 'last_time_for_rate',
            'loop_counter', 'current_median_temp', 'chamber_target', 'bed_target_temp',
            'start_chamber_temp', 'bed_min_temp', 'real_bed_temp', 'time_left_sec',
            'ema_chamber_temp', 'error_code']
    
    input_vars = {k: macro.get(k) for k in keys if k in macro}
    
    required = ['chamber_target', 'ema_chamber_temp', 'loop_counter', 'real_bed_temp']
    for key in required:
        if key not in input_vars:
            log.logger.error(f"missing required key: {key}")
            return
    
    result = process_heating(input_vars)
    
    if result:
        output_keys = [
            'bed_target_temp', 'error_code', 'heating_estimate_valid',
            'last_target_minutes', 'current_heat_rate_cpm',
            'wait_start_loop', 'last_time_for_rate'
        ]
        output = {k: result[k] for k in output_keys if k in result}
        if output:
            send_vars(output)
            log.logger.info(f"📤 отправлено {len(output)} переменных")

if __name__ == "__main__":
    main()
    