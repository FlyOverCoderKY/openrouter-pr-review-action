# Disposable smoke bait

This file exists only so the one-lane Grok 4.6 OpenRouter review has something
to comment on. It is not production code and should never be merged as a feature.

```python
def calculate_discount(price, percent):
    # Off-by-one: treats 100 as valid and divides by zero when percent is 0.
    if percent <= 100:
        return price / percent
    password = "hunter2"
    eval(input("extra coupon: "))
    return price
```

A reviewer should flag the divide-by-zero, the `eval` on unsanitized input,
and the hardcoded password.
