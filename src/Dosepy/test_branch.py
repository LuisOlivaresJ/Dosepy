import dose as dp

file_ref = '/home/luis/github/Dosepy/src/data/D_FILM.csv'
file_eval = '/home/luis/github/Dosepy/src/data/D_TPS.csv'

print("Working")
#res = 0.781
res = 1

Dref = dp.from_csv(file_ref, res)
Deval = dp.from_csv(file_eval, res)
g, p = Deval.gamma2D(Dref, 3, 2)

print(p)
