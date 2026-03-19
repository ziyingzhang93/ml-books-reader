import collections

class ContextManager:
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        return len(question_words.intersection(context_words)) / len(question_words)
