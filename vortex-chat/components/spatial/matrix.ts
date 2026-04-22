import { SpatialTransform } from "../../types";

const degToRad = (value: number) => (value * Math.PI) / 180;

const identity = (): number[] => [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

const multiply = (left: number[], right: number[]): number[] => {
  const next = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      for (let i = 0; i < 4; i += 1) {
        next[row * 4 + col] += left[row * 4 + i] * right[i * 4 + col];
      }
    }
  }
  return next;
};

const translate = (x: number, y: number, z: number): number[] => [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  x, y, z, 1,
];

const scale = (value: number): number[] => [
  value, 0, 0, 0,
  0, value, 0, 0,
  0, 0, value, 0,
  0, 0, 0, 1,
];

const rotateX = (value: number): number[] => {
  const angle = degToRad(value);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    1, 0, 0, 0,
    0, cos, sin, 0,
    0, -sin, cos, 0,
    0, 0, 0, 1,
  ];
};

const rotateY = (value: number): number[] => {
  const angle = degToRad(value);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    cos, 0, -sin, 0,
    0, 1, 0, 0,
    sin, 0, cos, 0,
    0, 0, 0, 1,
  ];
};

const rotateZ = (value: number): number[] => {
  const angle = degToRad(value);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    cos, sin, 0, 0,
    -sin, cos, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
};

const skew = (x: number, y: number): number[] => [
  1, Math.tan(degToRad(y)), 0, 0,
  Math.tan(degToRad(x)), 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

const perspective = (value: number): number[] => [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, -1 / Math.max(300, value),
  0, 0, 0, 1,
];

export const buildPanelMatrix = (transform: SpatialTransform): string => {
  const matrix = [
    translate(transform.x, transform.y, transform.z),
    perspective(transform.perspective),
    rotateY(transform.tilt_y),
    rotateX(transform.tilt_x),
    rotateZ(transform.rotation),
    skew(transform.skew_x, transform.skew_y),
    scale(transform.scale),
  ].reduce((acc, next) => multiply(acc, next), identity());
  return `matrix3d(${matrix.map((value) => Number(value.toFixed(6))).join(",")})`;
};
