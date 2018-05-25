import matplotlib.pyplot as plt

# 1.3

x1 = [10, 100, 1000, 10000]
y1 = open("train.txt", 'r+').read().splitlines()
for i in range(0,len(y1)):
    y1[i] = float(y1[i])
print y1
y2 = open("test.txt", 'r+').read().splitlines()
for i in range(0,len(y2)):
    y2[i] = float(y2[i])
print y2

plt.title('Average Log Likelihood across All the Sequences with Different Numbers of Sequences for Training')
plt.xlabel('Numbers of Sequences for Training')
plt.ylabel('Average Log Likelihood')

plt.plot(x1, y1, color="red", linewidth=2.5, linestyle="-", label="Training Dataset")
plt.plot(x1, y1, 'ro', color='black')
plt.plot(x1, y2, color="blue", linewidth=2.5, linestyle="-", label="Test Dataset")
plt.plot(x1, y2, 'ro', color='black')
plt.legend(loc='lower right')
plt.axis([0, 10000, -135, -90])
plt.show()
