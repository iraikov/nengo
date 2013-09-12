import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.core import Signal, Constant
from nengo.core import Direct, LIF, LIFRate
from nengo.tests.helpers import SimulatorTestCase, unittest, rms

import logging
logger = logging.getLogger(__name__)


class TestNonlinear(SimulatorTestCase):
    def test_direct(self):
        """Test direct mode"""

        d = 3
        n_steps = 3
        n_trials = 3

        rng = np.random.RandomState(seed=987)

        for i in xrange(n_trials):
            A = rng.normal(size=(d,d))
            fn = lambda x: np.cos(np.dot(A, x))

            x = np.random.normal(size=d)

            m = nengo.Model("")
            ins = m.add(Signal(n=d, name='ins'))
            pop = m.add(Direct(n_in=d, n_out=d, fn=fn))

            m._operators += [simulator.DotInc(Constant(np.eye(d)), ins, pop.input_signal)]
            m._operators += [simulator.ProdUpdate(Constant(np.eye(d)), pop.output_signal, Constant(0), ins)]

            sim = m.simulator(sim_class=self.Simulator)
            sim.signals[ins] = x

            p0 = np.zeros(d)
            s0 = np.array(x)
            for j in xrange(n_steps):
                tmp = p0
                p0 = fn(s0)
                s0 = tmp
                sim.step()
                assert np.allclose(s0, sim.signals[ins]), (s0,sim.signals[ins])
                assert np.allclose(p0, sim.signals[pop.output_signal]), (p0,sim.signals[pop.output_signal])

    def test_lif_builtin(self):
        """Test that the dynamic model approximately matches the rates

        N.B.: Uses the built-in lif equations, NOT the passed simulator
        """
        lif = LIF(10)
        J = np.arange(0, 5, .5)
        voltage = np.zeros(10)
        reftime = np.zeros(10)

        spikes = []
        spikes_ii = np.zeros(10)

        for ii in range(1000):
            lif.step_math0(.001, J + lif.bias, voltage, reftime, spikes_ii)
            spikes.append(spikes_ii.copy())

        sim_rates = np.sum(spikes, axis=0)
        math_rates = lif.rates(J)
        self.assertTrue(np.allclose(sim_rates, math_rates, atol=1, rtol=0.02))

    def test_lif(self):
        """Test that the dynamic model approximately matches the rates"""
        d = 1
        n = 5e3

        m = nengo.Model("")
        ins = m.add(Signal(n=d, name='ins'))
        lif = m.add(LIF(n))

        m._operators += [simulator.DotInc(Constant(np.ones((n,d))), #arbitrary encoders, doesn't really matter
                                          ins,
                                          lif.input_signal)]

        sim = m.simulator()
        sim.signals[ins] = np.random.normal(loc=0, scale=1, size=d)

        dt = sim.model.dt
        t_final = 1.
        t = dt * np.arange(np.round(t_final / dt))
        nt = len(t)

        spikes = []
        for ii in range(nt):
            sim.step()
            spikes.append(sim.signals[lif.output_signal])

        sim_rates = np.sum(spikes, axis=0) / t_final
        math_rates = lif.rates(sim.signals[lif.input_signal])
        logger.debug("ME = %f", (sim_rates - math_rates).mean())
        logger.debug("RMSE = %f", rms(sim_rates - math_rates) / rms(math_rates))
        self.assertTrue(np.allclose(sim_rates, math_rates, atol=1, rtol=0.02))

    def test_lif_rate(self):
        """Test that the simulator rate model matches the built in one"""
        d = 1
        n = 5e3

        m = nengo.Model("")
        ins = m.add(Signal(n=d, name='ins'))
        lif = m.add(LIFRate(n))

        m._operators += [simulator.DotInc(Constant(np.ones((n,d))), #arbitrary encoders, doesn't really matter
                                                  ins,
                                                  lif.input_signal)]

        sim = m.simulator()
        sim.signals[ins][...] = np.random.normal(loc=0, scale=1, size=d)
        sim.step()

        sim_rates = sim.signals[lif.output_signal]
        math_rates = lif.rates(sim.signals[lif.input_signal])
        self.assertTrue(np.allclose(sim_rates, math_rates, atol=1, rtol=0.02))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
