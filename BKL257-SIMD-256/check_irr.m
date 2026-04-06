q:=257;
F2 := FiniteField(2);
P<t> := PolynomialRing(F2);


for i :=1 to 100 do
    f := P!(t^257 + t^i + 1);
    fac_f := Factorization(f);
    print "i = ",i, "Factors := ",fac_f;
end for;
