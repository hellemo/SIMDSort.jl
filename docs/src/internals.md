# Internals

Docstrings of the internal functions and types, for reading the source.
Start with [Design](design.md) §10 for the structure.

```@autodocs
Modules = [SIMDSort]
Public = false
Filter = t -> !(t in (SIMDSort.orderwords, SIMDSort.has_orderwords, SIMDSort.instrumented, SIMDSort.print_report))
```
