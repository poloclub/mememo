/**
 * Type definitions
 */

export type DocumentRecordStreamData = [string, number[]];

export interface DocumentDBEntry {
  id: number;
  text: string;
}

export interface DocumentRecord {
  embedding: number[];
  id: number;
  text: string;
}

export interface SimpleEventMessage {
  message: string;
}

export type Mutable<Type> = {
  -readonly [Key in keyof Type]: Type[Key];
};

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RectPoint {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Size {
  width: number;
  height: number;
}
