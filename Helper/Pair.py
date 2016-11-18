class Pair():
	o1 = None
	o2 = None

	def __contains__(self, item):
		if item is self.o1 or item is self.o2:
			return True
		else:
			return False

	def setPair(self, o1, o2):
		self.o1 = o1
		self.o2 = o2

	def other(self, o):
		if o is self.o1:
			return self.o2
		elif o is self.o2:
			return self.o1
		else:
			print 'Bad access to Pair. Object not exist!'
			return None

	def __init__(self, o1=None, o2=None):
		self.o1 = o1
		self.o2 = o2
