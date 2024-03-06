class Node:
    def __init__(self, valence, arousal):
        self.valence = valence
        self.arousal = arousal


class EmotionCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []

    def get(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None

    def put(self, node):
        if len(self.queue) >= self.capacity:
            self.queue.pop(0)
        self.queue.append(node)

    def get_mean_valence(self):
        if not self.queue:
            return 0
        total_valence = sum(node.arousal for node in self.queue)
        return total_valence / len(self.queue)

    def get_mean_arousal(self):
        if not self.queue:
            return 0
        total_arousal = sum(node.arousal for node in self.queue)
        return total_arousal / len(self.queue)

    def get_valence_trend(self):
        if len(self.queue) < 2:
            return None

        mid = len(self.queue) // 2
        new_valence = sum(node.valence for node in self.queue[mid:]) / (len(self.queue) - mid)
        old_valence = sum(node.valence for node in self.queue[:mid]) / mid

        if new_valence > old_valence:
            return "Increasing"
        elif new_valence < old_valence:
            return "Decreasing"
        else:
            return "Stable"

    def get_arousal_trend(self):
        if len(self.queue) < 2:
            return None

        mid = len(self.queue) // 2
        new_arousal = sum(node.arousal for node in self.queue[mid:]) / (len(self.queue) - mid)
        old_arousal = sum(node.arousal for node in self.queue[:mid]) / mid

        if new_arousal > old_arousal:
            return "Increasing"
        elif new_arousal < old_arousal:
            return "Decreasing"
        else:
            return "Stable"


# 初始化容量为10的缓存实例
cache = EmotionCache(10)
