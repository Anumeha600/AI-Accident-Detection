/*
 * PHASE 6 -- MPU6050 I2C driver (STM32 HAL).
 * ---------------------------------------------------------------------
 * VERIFICATION STATUS: written and statically reviewed only. This has
 * NOT been compiled or tested on real STM32 hardware -- there is no
 * physical board available in this environment. Test on your own
 * hardware before trusting it (see the Phase 6 write-up for how).
 *
 * Assumes STM32 HAL (any family that ships stm32xxxx_hal.h with the
 * standard HAL_I2C_Mem_Read/Write API -- adjust the #include below to
 * match your CubeIDE-generated project, e.g. stm32f1xx_hal.h for an
 * STM32F1 board, stm32l4xx_hal.h for an STM32L4 board, etc.). This
 * project assumed a common STM32 Nucleo-F4-series board as a default
 * since no specific board was identified in the existing project --
 * confirm/adjust for your actual board.
 */

#ifndef MPU6050_H
#define MPU6050_H

#include "stm32f4xx_hal.h"   /* CHANGE to match your board's HAL family header */
#include <stdint.h>

/* 7-bit I2C address 0x68 when the MPU6050's AD0 pin is tied to GND
 * (the common default). HAL's Mem_Read/Write functions want it
 * pre-shifted left by 1 (the 8-bit form that reserves bit0 for R/W). */
#define MPU6050_I2C_ADDR         (0x68 << 1)
/* If AD0 is tied to VCC instead, use (0x69 << 1) here. */

#define MPU6050_REG_PWR_MGMT_1   0x6B
#define MPU6050_REG_WHO_AM_I     0x75
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_GYRO_CONFIG  0x1B
#define MPU6050_REG_ACCEL_XOUT_H 0x3B

#define MPU6050_WHO_AM_I_EXPECTED 0x68

/* Scale factors for the default power-on full-scale ranges this driver
 * configures: +/-2g (ACCEL_CONFIG=0x00) and +/-250 deg/s
 * (GYRO_CONFIG=0x00). Per the MPU6050 register map datasheet. */
#define MPU6050_ACCEL_SCALE_LSB_PER_G   16384.0f
#define MPU6050_GYRO_SCALE_LSB_PER_DPS  131.0f

typedef struct {
    float accel_x_g;
    float accel_y_g;
    float accel_z_g;
    float gyro_x_dps;
    float gyro_y_dps;
    float gyro_z_dps;
} MPU6050_Reading;

/* Confirms the sensor responds with the expected WHO_AM_I value, wakes
 * it from sleep mode, and sets the accel/gyro ranges explicitly.
 * Returns HAL_OK on success, HAL_ERROR/other HAL status on failure. */
HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c);

/* Burst-reads the 14 accel+temp+gyro registers in one I2C transaction
 * and fills `out` with the converted g / deg-per-second values. */
HAL_StatusTypeDef MPU6050_ReadScaled(I2C_HandleTypeDef *hi2c, MPU6050_Reading *out);

#endif /* MPU6050_H */
