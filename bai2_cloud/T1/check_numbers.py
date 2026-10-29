"""Numbers quoted in novelty_audit.md. Pure arithmetic, no data. Run: python3 check_numbers.py"""
import json
import math

out = {}
# Outline, Proposition 1(a) example: r = 1e-4, k_R = 5000.
r, k_R = 1e-4, 5000
out["prop1a_exact_(1-r)^kR"] = (1 - r) ** k_R
out["prop1a_bound_exp(-r*kR)"] = math.exp(-r * k_R)
# CURE defaults (cure_spec.md sec. 1): B = 1000, rho = 0.75; Bai 1 harness k = 2000.
for B in (1000, 2000):
    B_S = max(1, int(round(B * 0.75)))
    out[f"cure_B{B}_short_long"] = [B_S, B - B_S]
# Outline: k_H <= 0.1k, evaluated at the Bai 1 harness size k = 2000.
out["kH_cap_at_k2000"] = 0.1 * 2000
print(json.dumps(out, indent=2))
with open("check_numbers_out.json", "w") as f:
    json.dump(out, f, indent=2)
