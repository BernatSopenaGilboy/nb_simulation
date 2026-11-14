# Questions to ask

## Observability of belief-network weights vs beliefs

Is it realistic to assume agents can see the whole weight structure (`w`) of their contacts but not those contacts’ other beliefs?

Belief structure seems harder to “see” than opinions. Proposed change: weights can only be read from connections to the **focal** belief, and only **H** of those edges at a time, where **H** is a parameter (sampled at random each update).

## Hebbian updates when only the focal belief is socially driven

Why run Hebbian updates over the full belief system if the only belief that changes through social / external coupling is the focal belief?

Wouldn’t that process mostly reinforce the starting weights indefinitely? What is the point of adapting non-focal edges under that asymmetry?
