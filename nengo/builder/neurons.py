import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.dists import Distribution
from nengo.exceptions import BuildError
from nengo.neurons import NeuronType
from nengo.utils.numpy import is_array_like


class SimNeurons(Operator):
    """Set a neuron model output for the given input current.

    Implements ``neurons.step(dt, J, output, **state)``.

    Parameters
    ----------
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step`` function.
    J : Signal
        The input current.
    output : Signal
        The neuron output signal that will be set.
    state : list, optional
        A list of additional neuron state signals set by ``step``.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    J : Signal
        The input current.
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step`` function.
    output : Signal
        The neuron output signal that will be set.
    state : list
        A list of additional neuron state signals set by ``step``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] + state``
    2. incs ``[]``
    3. reads ``[J]``
    4. updates ``[]``
    """

    def __init__(self, neurons, J, output, state=None, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons

        self.sets = [output]
        self.incs = []
        self.reads = [J]
        self.updates = []

        self.state = {} if state is None else state
        self.sets.extend(self.state.values())

    @property
    def J(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def _descstr(self):
        return "%s, %s, %s" % (self.neurons, self.J, self.output)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        state = {name: signals[sig] for name, sig in self.state.items()}

        def step_simneurons():
            self.neurons.step(dt, J, output, **state)

        return step_simneurons


@Builder.register(NeuronType)
def build_neurons(model, neurontype, neurons):
    """Builds a `.NeuronType` object into a model.

    This function adds a `.SimNeurons` operator connecting the input current to the
    neural output signals, and handles any additional state variables defined
    within the neuron type.

    Parameters
    ----------
    model : Model
        The model to build into.
    neurontype : NeuronType
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.NeuronType` instance.
    """
    dtype = model.sig[neurons]["in"].dtype
    n_neurons = neurons.size_in
    state_init = neurontype.make_state(n_neurons, model.dt, dtype=dtype)
    state = {}

    for key, init in state_init.items():
        if key in model.sig[neurons]:
            raise BuildError("State name %r overlaps with existing signal name" % key)

        if isinstance(init, Distribution):
            raise NotImplementedError()

        if not is_array_like(init):
            raise BuildError("State init must be a distribution or array-like")

        if init.ndim == 1 and init.size != n_neurons or init.ndim > 1:
            raise BuildError(
                "State init array must be 0-D, or 1-D of length `n_neurons`"
            )

        model.sig[neurons][key] = Signal(
            initial_value=init, name="%s.%s" % (neurons, key)
        )
        state[key] = model.sig[neurons][key]

    model.add_op(
        SimNeurons(
            neurons=neurontype,
            J=model.sig[neurons]["in"],
            output=model.sig[neurons]["out"],
            state=state,
        )
    )
