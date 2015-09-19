
__all__ = ['ReiterableMixin']


class Reiterable:
    def __init__(self, func, docs):
        self.func = func
        self.docs = docs

    def __hash__(self):
        return hash((self.docs, self.func))

    def __eq__(self, other):
        return self.docs == other.docs and self.func == other.func

    def __iter__(self):
        yield from self.func(self.docs)


class ReiterableMixin:
    # noinspection PyUnresolvedReferences
    def transform(self, docs):
        return Reiterable(self._transform, docs)
