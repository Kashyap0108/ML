import matplotlib.pyplot as plt
f = int(input("Enter the frequency"))
c = 3*pow(10,8)
pi = 3.14
d = int(input("Enter the distance"))
result = []
distance = []
for i in range(0,d,2):
  FSPL = ((4*pi*f*i)/c)**2
  result.append(FSPL)
  distance.append(i)
print(f"The distance range is from 0 to {d}")
plt.plot(distance,result)
plt.xlabel("Distance")
plt.ylabel("FSPL")
plt.title("Free Space Path Loss vs Distance")
