import sys, time, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_build():
    logger.info('Test 1: building IDAES SOEC environment ...')
    from envs.soec_env_idaes import SOECEnvIDEAS
    env = SOECEnvIDEAS()
    logger.info('  Build OK')
    return env

def test_step(env):
    logger.info('Test 2: reset and step ...')
    obs, _ = env.reset()
    assert obs.shape == (5,), f'Expected (5,), got {obs.shape}'
    v_cell, dT, util, re_now, re_pred = obs
    logger.info(f'  obs: v={v_cell:.3f}V dT={dT:+.1f}K util={util:.2f} RE={re_now:.2f}')
    t0 = time.time()
    obs2, rew, term, trunc, info = env.step([0.0])
    elapsed = time.time() - t0
    vc=info["v_cell"]; le=info["load_error"]; logger.info(f"  step: {elapsed*1000:.0f}ms  v={vc:.3f}V  load_err={le:.4f}")
    logger.info(f'  Estimated 100K steps: {elapsed*100000/3600:.1f} hours')
    assert not info['safety_violation'], 'Safety violation at nominal'
    assert elapsed < 10.0, f'Too slow: {elapsed:.1f}s'
    logger.info('  OK')

def test_physics(env):
    logger.info('Test 3: physics response ...')
    env.reset()
    _, _, _, _, info_lo = env.step([-0.05])
    _, _, _, _, info_hi = env.step([ 0.05])
    u=info_lo["util"]; v=info_lo["v_cell"]; logger.info(f"  util_lo={u:.2f} v={v:.4f}V")
    u=info_hi["util"]; v=info_hi["v_cell"]; logger.info(f"  util_hi={u:.2f} v={v:.4f}V")
    logger.info('  OK')

if __name__ == '__main__':
    logger.info('IDAES SOEC Smoke Test')
    try:
        env = test_build()
        test_step(env)
        test_physics(env)
        logger.info('All tests passed')
    except Exception as e:
        logger.error('FAILED: %s', e)
        import traceback; traceback.print_exc()
        import sys; sys.exit(1)