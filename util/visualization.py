def logit_to_point(input_dim, output_dim, input_x, input_y):
    """
    Maps coordinates from the output dimension space back to the input dimension space.

    Args:
        input_dim (tuple): (height, width) of the input image.
        output_dim (tuple): (height, width) of the output (e.g., logits).
        input_x (int or float): x-coordinate in output space.
        input_y (int or float): y-coordinate in output space.

    Returns:
        tuple: (mapped_x, mapped_y) in input space.
    """
    in_h, in_w = input_dim
    out_h, out_w = output_dim

    mapped_x = round(input_x * in_w / out_w)
    mapped_y = round(input_y * in_h / out_h)
    return mapped_x, mapped_y
