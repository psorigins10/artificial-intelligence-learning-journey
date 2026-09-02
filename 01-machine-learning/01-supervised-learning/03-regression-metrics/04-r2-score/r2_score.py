from sklearn.metrics import r2_score

y_actual = [3, 5, 7, 9, 11]
y_predicted = [2.5, 5, 7.5, 8.5, 10]

r2 = r2_score(y_actual, y_predicted)

print("R² Score:", r2)