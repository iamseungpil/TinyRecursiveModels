# HelmARC DSL Primitives Reference

**Generated**: 2025-10-25
**Total Primitives**: 60
**Source**: HelmARC dataset analysis

---

## ⚠️ Important Corrections

### flipx vs flipy (CORRECTED)

Based on actual behavior testing:

| Primitive | **CORRECT Definition** | Grid Transform Example |
|-----------|------------------------|------------------------|
| `flipx` | Flips **vertically** (upside-down) | [[1,2],[3,4]] → [[3,4],[1,2]] |
| `flipy` | Flips **horizontally** (left-right) | [[1,2],[3,4]] → [[2,1],[4,3]] |

**Previous incorrect descriptions** (found in GPT-OSS generated data):
- ❌ flipx: "horizontally, swapping left and right" (WRONG)
- ❌ flipy: "vertically, swapping top and bottom" (WRONG)

---

## Primitive Categories

### 1. Rotation Primitives

| Primitive | Description |
|-----------|-------------|
| `rot90` | Rotates the grid 90 degrees clockwise |
| `rot180` | Rotates the grid 180 degrees |
| `rot270` | Rotates the grid 270 degrees clockwise (or 90 degrees counter-clockwise) |
| `swapxy` | Transposes the grid, swapping rows and columns (diagonal flip) |

---

### 2. Flip Primitives

| Primitive | Description | Axis | Example |
|-----------|-------------|------|---------|
| `flipx` | Flips the grid **vertically** (upside-down), swapping top and bottom rows | x-axis | [[1,2],[3,4]] → [[3,4],[1,2]] |
| `flipy` | Flips the grid **horizontally** (left-right), swapping left and right columns | y-axis | [[1,2],[3,4]] → [[2,1],[4,3]] |

---

### 3. Mirror Primitives (Expand by Mirroring)

| Primitive | Description | Example |
|-----------|-------------|---------|
| `mirror` | Creates a mirrored copy of the grid | General mirroring |
| `mirrorX` | Extends the grid by mirroring **horizontally** (creates left-right symmetry) | [[1,2],[3,4]] → [[1,2,2,1],[3,4,4,3]] |
| `mirrorY` | Extends the grid by mirroring **vertically** (creates top-bottom symmetry) | [[1,2],[3,4]] → [[1,2],[3,4],[3,4],[1,2]] |

**Note**: `mirrorX`/`mirrorY` **extend** the grid, while `flipx`/`flipy` **transform** in-place.

---

### 4. Gravity Primitives

| Primitive | Description |
|-----------|-------------|
| `gravity_left` | Moves all colored cells to the left within each row, like gravity pulling leftward |
| `gravity_right` | Moves all colored cells to the right within each row, like gravity pulling rightward |
| `gravity_up` | Moves all colored cells upward within each column, like gravity pulling upward |
| `gravity_down` | Moves all colored cells downward within each column, like gravity pulling downward |

---

### 5. Half Selection Primitives

| Primitive | Description |
|-----------|-------------|
| `top_half` | Returns the top half of the grid |
| `bottom_half` | Returns the bottom half of the grid |
| `left_half` | Returns the left half of the grid |
| `right_half` | Returns the right half of the grid |

---

### 6. Color Operations

| Primitive | Description |
|-----------|-------------|
| `colour` | Returns or identifies color values |
| `count` | Counts occurrences of a pattern or color |
| `get_bg` | Identifies and returns cells matching the background color |
| `rarestcol` | Finds the rarest (least frequent) color in the grid |
| `set_bg` | Sets the background color to a specified value |
| `setcol` | Sets all cells to a specific color |
| `topcol` | Identifies the color of the topmost non-zero cell |

---

### 7. IC (Image Correction / Connected Components) Primitives

#### 7.1. Basic IC Operations

| Primitive | Description |
|-----------|-------------|
| `ic_center` | Centers the colored pattern within the grid |
| `ic_compress2` | Removes empty rows and columns to compress the grid (level 2) |
| `ic_compress3` | Removes empty rows and columns to compress the grid (level 3) |
| `ic_connect` | Connects separated components of the same color |
| `ic_composegrowing` | Composes patterns with growing size |
| `ic_embed` | Embeds one grid pattern into another |
| `ic_erasecol` | Removes all cells of a specific color (replaces with black/0) |
| `ic_fill` | Fills interior empty cells within colored regions |
| `ic_filtercol` | Filters cells by color, keeping only specified colors |
| `ic_interior` | Extracts only the interior cells of colored regions |
| `ic_invert` | Inverts the color pattern |
| `ic_makeborder` | Creates a border around colored regions |
| `ic_pickunique` | Selects unique color patterns from the grid |
| `ic_toorigin` | Moves all colored cells to the origin (top-left) |

#### 7.2. IC Split Operations (Split into Subgrids)

| Primitive | Description |
|-----------|-------------|
| `ic_splitall` | Splits the grid into all connected components |
| `ic_splitcols` | Splits by color groups |
| `ic_splitcolumns` | Splits the grid by columns into separate subgrids |
| `ic_splitrows` | Splits the grid by rows into separate subgrids |

---

### 8. Pick/Select Primitives (Choose from Multiple Options)

| Primitive | Description |
|-----------|-------------|
| `pickcommon` | Selects the most common pattern from multiple options |
| `pickmax_cols` | Selects columns with maximum values |
| `pickmax_count` | Selects the component with maximum count |
| `pickmax_interior_count` | Selects component with maximum interior cell count |
| `pickmax_neg_count` | Selects component with maximum negative count |
| `pickmax_neg_interior_count` | Selects component with maximum negative interior count |
| `pickmax_neg_size` | Selects component with maximum negative size metric |
| `pickmax_size` | Selects the largest component by size |
| `pickmax_x_neg` | Selects component with minimum x position |
| `pickmax_x_pos` | Selects component with maximum x position |
| `pickmax_y_neg` | Selects component with minimum y position |
| `pickmax_y_pos` | Selects component with maximum y position |

---

### 9. Repeat Primitives

| Primitive | Description |
|-----------|-------------|
| `repeat` | Repeats the pattern multiple times |

---

### 10. Higher-Order Functions

| Primitive | Description |
|-----------|-------------|
| `lcons` | List constructor operation |
| `map` | Applies a function to each element |
| `mklist` | Creates a list from elements |

---

### 11. Composition Operations

| Primitive | Description |
|-----------|-------------|
| `fillobj` | Fills objects with a specific pattern |
| `overlay` | Overlays one grid pattern on top of another |

---

### 12. Logical Operations

| Primitive | Description |
|-----------|-------------|
| `logical_and` | Performs logical AND operation between two grids |

---

### 13. Utility Operations

| Primitive | Description |
|-----------|-------------|
| `split8` | Splits the grid into 8 parts |

---

## Usage in HelmARC Dataset

### Most Frequently Used Primitives

Based on 8,572 HelmARC samples:

1. **IC operations** (ic_fill, ic_erasecol, ic_compress, etc.) - Most common
2. **Gravity operations** (gravity_left, gravity_right, etc.)
3. **Flip/Mirror operations** (flipx, flipy, mirrorX, mirrorY)
4. **Rotation operations** (rot90, rot180, rot270)
5. **Pick operations** (pickmax_size, pickmax_count, etc.)

### Composition Examples

Primitives can be composed to create complex transformations:

```
Simple (1-step):
(lambda (flipx $0))

Composed (2-step):
(lambda (gravity_down (flipx $0)))

Complex (3-step):
(lambda (ic_fill (ic_compress2 (pickmax_interior_count (ic_splitall $0)))))
```

---

## Mathematical Definitions

### Coordinate System

- **x-axis**: Vertical (top-to-bottom)
- **y-axis**: Horizontal (left-to-right)

### Flip Operations (Mathematical)

- **flipx**: Reflection across x-axis → flips vertically (upside-down)
- **flipy**: Reflection across y-axis → flips horizontally (left-right)

This follows standard mathematical convention where:
- Flipping across x-axis affects y-coordinates (vertical flip)
- Flipping across y-axis affects x-coordinates (horizontal flip)

---

## Implementation Source

All primitives are implemented in:
```
/home/ubuntu/dreamcoder-arc/ec/dreamcoder/domains/arc/arcPrimitivesIC2.py
```

Total registered primitives in implementation: **81**
Used in HelmARC dataset: **60**

---

## Known Issues in Generated Data

### ❌ Incorrect Descriptions (Affecting 48.7% of dataset)

The GPT-OSS generated descriptions for flipx/flipy had **reversed** definitions:

| Samples Affected | Description |
|------------------|-------------|
| 612 TYPE 1 samples | flipx described as "horizontal" (should be vertical) |
| 591 TYPE 1 samples | flipy described as "vertical" (should be horizontal) |
| 5,943 TYPE 2 samples | flipx in wrong/correct programs |
| 920 TYPE 2 samples | flipy in wrong/correct programs |
| **Total: 8,066 samples (48.7%)** | Need correction |

### ✅ Correction Options

1. **Full regeneration** with corrected prompt (~14 hours)
2. **Selective regeneration** of affected samples (~7 hours)
3. **Post-processing text replacement** (immediate, but requires validation)

---

## References

- **Dataset**: HelmARC (Helmholtz-sampled ARC problems)
- **Total Samples**: 8,572
- **Original ARC**: [https://github.com/fchollet/ARC](https://github.com/fchollet/ARC)
- **DreamCoder**: [https://github.com/ellisk42/ec](https://github.com/ellisk42/ec)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Status**: ⚠️ flipx/flipy definitions corrected from original GPT-OSS generation
