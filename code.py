    Then:
        x_standard = P_B @ x_B
        x_B        = inverse(P_B) @ x_standard

    For a linear transformation with standard matrix A, and bases B (domain)
    and C (codomain):
        [T]_(C <- B) = inverse(P_C) @ A @ P_B
    """
    # Basis B and basis C are represented by their column vectors.
    P_B = np.array([
        [1, 1],
        [1, -1]
    ], dtype=float)

    P_C = np.array([
        [2, 0],
        [0, 1]
    ], dtype=float)

    # A vector expressed in basis B coordinates.
    vector_B = np.array([3, 2], dtype=float)

    # Convert from B-coordinates to standard coordinates.
    vector_standard = P_B @ vector_B

    # Convert the standard-coordinate vector into C-coordinates.
    vector_C = np.linalg.solve(P_C, vector_standard)

    print("Basis matrix P_B:")
    print(P_B)
    print("\nBasis matrix P_C:")
    print(P_C)
    print("\nVector coordinates in basis B:")
    print(vector_B)
    print("\nVector in standard coordinates (P_B @ vector_B):")
    print(vector_standard)
    print("\nSame vector expressed in basis C:")
    print(vector_C)

    # Confirm that converting back from C gives the original vector.
    assert np.allclose(P_C @ vector_C, vector_standard)
    print("\nVector change-of-basis verification: VERIFIED")

    # Define a linear transformation in standard coordinates.
    A = np.array([
        [2, 1],
        [1, 3]
    ], dtype=float)

    # Matrix of T from basis B coordinates to basis C coordinates.
    # solve(P_C, A @ P_B) is numerically preferable to explicitly
    # calculating inverse(P_C) @ A @ P_B.
    transformation_C_from_B = np.linalg.solve(P_C, A @ P_B)

    print("\nStandard matrix A of the linear transformation:")
    print(A)
    print("\nTransformation matrix [T]_(C <- B):")
    print(transformation_C_from_B)

    # Verify the transformation using a sample coordinate vector.
    result_via_new_basis = transformation_C_from_B @ vector_B
    result_standard = A @ vector_standard
    result_converted_to_C = np.linalg.solve(P_C, result_standard)

    assert np.allclose(result_via_new_basis, result_converted_to_C)
    print("\nLinear transformation change-of-basis verification: VERIFIED")


def main():
    """Run all linear algebra demonstrations."""
    section("1. VECTOR SPACE DIMENSIONS AND RANK-NULLITY THEOREM")

    # A custom 3 x 4 matrix. Its rank and nullity describe the
    # dimensions of its column space and null space, respectively.
    matrix_A = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [1, 1, 1, 1]
    ], dtype=float)

    analyze_vector_space(matrix_A)

    section("2. EIGENVALUES AND EIGENVECTORS")

    # A custom 2 x 2 square matrix for eigenvalue computation.
    matrix_B = np.array([
        [4, 2],
        [1, 3]
    ], dtype=float)

    calculate_eigenvalues(matrix_B)

    section("3. CHANGE OF BASIS TRANSFORMATIONS")
    change_of_basis()

    section("ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("NumPy linear algebra demonstrations and checks completed.")


if __name__ == "__main__":
    main()
