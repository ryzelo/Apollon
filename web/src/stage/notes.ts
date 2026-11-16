// The plain-language explanation every view carries: a title, a sentence or two, and how to read it.
import type { StageView } from "../store";

export type Note = { title: string; text: string; reading: [string, string][] };

/** The space view shows a different scene for every analysis. */
export const SPACE_NOTES: Record<string, Note> = {
  decompose: {
    title: "The model in space",
    text: "Layers run from left to right, token positions from front to back. Height is what a part adds to the target token, or takes from it. Drag to turn, scroll to zoom.",
    reading: [["Place", "Its layer and the token position it works at."], ["Height", "Signed change of the target’s log-odds, on the compressed scale of the axis."], ["Lanes", "MLPs and other types stand behind the last token. Attention reads along the bright row."], ["Arcs", "What a head carries from a position it reads. Floor lines show where it looks."], ["Dim", "Units of a remainder, not measured on their own."]],
  },
  probe: {
    title: "One component, unit by unit",
    text: "The plane on the left holds every unit, at the height of what its removal costs the target. From the strongest units a path runs through the later layers.",
    reading: [["Plane", "One point per unit, most supporting at the back."], ["Height", "Target log-odds with the unit minus without it."], ["White", "The same difference, read out after every later layer."], ["Colour", "A later component that votes differently without the unit."]],
  },
  trace: {
    title: "What feeds this neuron",
    text: "The white point is the traced neuron. Each line leads back to a source of its activation: an earlier neuron, a head with the positions it reads, or the input.",
    reading: [["Place", "The source’s layer and token position."], ["Height", "Its share of the activation of the node it feeds."], ["Colour", "Neurons orange, heads and what they read blue, input white."], ["Depth", "Only the strongest sources are followed at every step."]],
  },
  circuit: {
    title: "The circuit in space",
    text: "Layers run from left to right, token positions from front to back. The clouds on the floor show where scalar products stay switched on. Above them stand the leaves of the circuit. Drag to turn, scroll to zoom.",
    reading: [["Floor", "Kept atoms per layer and position, denser where more are kept."], ["Height", "Target log-odds in the circuit minus without the leaf, removed alone."], ["Colour", "Attention blue, MLP orange, per-layer embedding violet."], ["Arcs", "Leaves that matter most, joined to the target."]],
  },
  neuron: {
    title: "A neuron from its weights",
    text: "No prompt is involved. On the left is what the neuron reads, on the right what it writes into. Height and depth are the two numbers a source adds to the neuron’s computation.",
    reading: [["Place", "The layer of the source or the target."], ["Height", "p, the part that enters the activation directly."], ["Depth", "q, the part that enters through the gate."], ["Lines", "Stronger connections draw brighter."]],
  },
};

export const NOTES: Record<Exclude<StageView, "space">, Note> = {
  flow: {
    title: "From every layer to the answer",
    text: "Each line carries one component’s contribution to the target token. Lines that raise it run to the upper point, lines that lower it to the lower one.",
    reading: [["Place", "Its layer, from input on the left to the last layer on the right."], ["Row", "The type of component."], ["Lines", "Stronger effects draw brighter, wider lines."], ["Dots", "Its measured parts: heads, neurons or sources. One dot is one part."]],
  },
  units: {
    title: "Where the effect sits",
    text: "Every box is one component, sized by the size of its effect. The points inside it are its units: heads, neurons or sources.",
    reading: [["Area", "The absolute effect on the target."], ["Colour", "The type of component."], ["Points", "One per unit. Bright ones were measured on their own; dim ones belong to the remainder, at most 3,000 drawn per box."], ["Faint", "The component lowers the target."]],
  },
  table: {
    title: "Every measurement",
    text: "All components of this run with their direct contribution and their total effect.",
    reading: [["Direct", "Read out through the final norm on its own."], ["Total", "Change when the component is removed and the model reruns."]],
  },
};

export const CHAIN_NOTES: Record<string, Note> = {
  trace: {
    title: "What drives this neuron",
    text: "Each source feeds the neuron’s activation. The neuron then raises or lowers the target token, shown by the white line to the right.",
    reading: [["Place", "The layer the source sits in."], ["Lines", "Brighter lines add more to the activation."], ["Remainder", "All sources too small to show on their own."], ["White", "The neuron’s own effect on the target."]],
  },
  probe: {
    title: "Every unit of one component",
    text: "Each unit was removed on its own and the model rerun. The column ranks units from the largest gain to the largest loss for the target.",
    reading: [["Order", "From most supporting to most suppressing."], ["Lines", "Brighter lines change the target more."], ["White", "The whole component’s effect on the target."]],
  },
};
