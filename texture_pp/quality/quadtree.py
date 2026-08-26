"""Quadtree-based mask refinement."""

from __future__ import annotations

import numpy as np


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class QuadTreeNode:
    def __init__(self, rect):
        self.rect = rect
        self.children = []
        self.is_leaf = True
        self.data = None
        self.status = "UNDEFINED"


def recursive_decompose(node, mask, min_block_size):
    x, y, w, h = node.rect.x, node.rect.y, node.rect.w, node.rect.h
    if (mask[y:y + h, x:x + w] == 1).all():
        node.is_leaf = True
        node.data = mask[y:y + h, x:x + w]
        node.status = "SR"
        return
    elif (mask[y:y + h, x:x + w] == 0).all():
        node.is_leaf = True
        node.data = mask[y:y + h, x:x + w]
        node.status = "KEEP"
        return
    if w <= min_block_size or h <= min_block_size:
        node.is_leaf = True
        node.data = mask[y:y + h, x:x + w]
        node.status = "SR"
        return
    node.is_leaf = False
    node.status = "MIXED"
    node.data = mask[y:y + h, x:x + w]
    half_w = w // 2
    half_h = h // 2
    node.children.append(QuadTreeNode(Rect(x, y, half_w, half_h)))
    node.children.append(QuadTreeNode(Rect(x + half_w, y, half_w, half_h)))
    node.children.append(QuadTreeNode(Rect(x, y + half_h, half_w, half_h)))
    node.children.append(QuadTreeNode(Rect(x + half_w, y + half_h, half_w, half_h)))
    for child in node.children:
        recursive_decompose(child, mask, min_block_size)


def get_mask(node, mask):
    if node.is_leaf:
        if node.status == "SR":
            x, y, w, h = node.rect.x, node.rect.y, node.rect.w, node.rect.h
            mask[y:y + h, x:x + w] = 1
        return mask
    else:
        for child in node.children:
            new_mask = get_mask(child, mask)
        return new_mask


def get_new_mask(mask, min_block_size: int = 9) -> np.ndarray:
    h, w = mask.shape
    root = QuadTreeNode(Rect(0, 0, w, h))
    recursive_decompose(root, mask, min_block_size)
    new_mask = np.zeros_like(mask)
    new_mask = get_mask(root, new_mask)
    return new_mask
