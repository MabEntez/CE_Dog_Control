import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("shepard_dog_prevalence.csv")

# 2. Extract only the last 63 entries
# .copy() is used to ensure we are working with a clean slice of data
data_tail = df.tail(120).copy()
yearly_data = data_tail.groupby(np.arange(len(data_tail)) // 12).mean()

# 3. Perform the calculation: (Spray - 0.16) / Deworm_4
# Note: This assumes 0.16 is your target or baseline constant
yearly_data["Prop_spray"] = (yearly_data["No_int_Prev"] - yearly_data["Spray_Prev"]) / yearly_data["No_int_Prev"]
yearly_data["Prop_norm"] = (yearly_data["No_int_Prev"] - yearly_data["Deworm_4_Prev"]) / yearly_data["No_int_Prev"]
yearly_data["Prop_8"] = (yearly_data["No_int_Prev"] - yearly_data["Deworm_8_Prev"]) / yearly_data["No_int_Prev"]

# 4. View the results
print(yearly_data[["Month", "No_int_Prev", "Spray_Prev", "Deworm_4_Prev", "Prop_spray", "Prop_norm"]])

L = np.average([73,77,53,66,92,120,112])
y0 = np.average([50,24,31,23,13,37,29])
k = np.float64(16.08458)
x0 = np.float64(7.929528)
years = np.array(range(0,10))

y = ((L - y0 )/ (1 + np.exp(k * (years - x0))) + y0)

Spray_HI = y * (yearly_data["Spray_Prev"] / 0.4) * (1 - yearly_data["Prop_spray"])
print(Spray_HI)

Norm_HI = y * (yearly_data["Deworm_4_Prev"] / 0.4) * (1 - yearly_data["Prop_norm"])
print(Norm_HI)

Norm_8_HI = y * (yearly_data["Deworm_8_Prev"] / 0.4) * (1 - yearly_data["Prop_8"])
print(Norm_8_HI)

x = range(0, len(Spray_HI))
plt.figure(figsize=(8, 5))
plt.plot(x, Norm_HI, color="black", alpha=0.8, linewidth=2, label="Deworming (4 times annually)")
plt.plot(x, Norm_8_HI, color="blue", alpha=0.6, linewidth=2, label="Deworming (8 times annually)")
plt.plot(x, Spray_HI, color="red", alpha=0.6, linewidth=2, label="Deworming (8 times annually) + Spraying")

plt.xlabel("Number of Years Post Intervention")
plt.ylabel("Human CE incidence per 100,000 (Year Average)")
plt.title("Human Incidence Estimated from Dog Prevalence")
plt.legend()
plt.tight_layout()
plt.savefig('Human_incidence_8.png', dpi=300, bbox_inches='tight')
plt.show()