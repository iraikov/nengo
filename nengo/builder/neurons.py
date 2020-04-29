from nengo.builder import Builder, Operator, Signal
from nengo.dists import Distribution
from nengo.exceptions import BuildError
from nengo.neurons import NeuronType
from nengo.rc import rc
from nengo.utils.numpy import is_array_like


class SimNeurons(Operator):
    """Set a neuron model output for the given input current.

    Implements ``neurons.step(dt, J, **state)``.

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

    def __init__(self, neurons, J, state=None, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons

        self.sets = []
        self.incs = []
        self.reads = [J]
        self.updates = []

        self.state = {}
        if state is not None:
            for name, sig in state.items():
                # The signals actually stored in `self.sets` can be modified by the
                # optimizer. To allow this possibility, we store the index of the
                # signal in the sets list instead of storing the signal itself.
                self.state[name] = len(self.sets)
                self.sets.append(sig)

    @property
    def J(self):
        return self.reads[0]

    @property
    def _descstr(self):
        return "%s, %s" % (self.neurons, self.J)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        state = {name: signals[self.sets[idx]] for name, idx in self.state.items()}

        def step_simneurons():
            self.neurons.step(dt, J, **state)

        return step_simneurons


def get_neuron_states(model, neurontype, neurons, dtype=None):
    """Get the initial neuron state values for the neuron type."""
    dtype = rc.float_dtype if dtype is None else dtype
    ens = neurons.ensemble
    rng = np.random.RandomState(model.seeds[ens] + 1)
    n_neurons = neurons.size_in

    if isinstance(ens.initial_phase, Distribution):
        phases = ens.initial_phase.sample(n_neurons, rng=rng)
    else:
        phases = ens.initial_phase

    if phases.ndim != 1 or phases.size != n_neurons:
        raise BuildError("`initial_phase` array must be 1-D of length `n_neurons`")

    return neurontype.make_neuron_state(phases, model.dt, dtype=dtype)


def sample_state_init(state_init, n_neurons, dtype=None):
    """Sample a state init value, ensuring the correct size. """
    dtype = rc.float_dtype if dtype is None else dtype

    if isinstance(state_init, Distribution):
        raise NotImplementedError()

    if not is_array_like(state_init):
        raise BuildError("State init must be a distribution or array-like")

    return state_init


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
    n_neurons = neurons.size_in
    state_init = get_neuron_states(model, neurontype, neurons)
    state = {}

    for key, init in state_init.items():
        if key in model.sig[neurons]:
            raise BuildError("State name %r overlaps with existing signal name" % key)

        state[key] = model.sig[neurons][key] = Signal(
            initial_value=sample_state_init(init, n_neurons),
            name="%s.%s" % (neurons, key),
        )

    model.sig[neurons]["out"] = state[neurontype.state[0]]
    model.add_op(
        SimNeurons(neurons=neurontype, J=model.sig[neurons]["in"], state=state)
    )
