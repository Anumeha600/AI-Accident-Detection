/*
 * PHASE 6 -- MPU6050 I2C driver implementation (STM32 HAL).
 * ---------------------------------------------------------------------
 * VERIFICATION STATUS: written and statically reviewed only. NOT
 * compiled or run on real hardware in this environment -- see
 * mpu6050.h and the Phase 6 write-up for details and how to test it
 * yourself once wired up.
 */

#include "mpu6050.h"

HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef status;
    uint8_t who_am_i = 0;
    uint8_t zero = 0x00;

    /* 1. Confirm the sensor is present and responding at the expected
     *    I2C address before doing anything else -- catches wiring or
     *    address mistakes early instead of silently streaming garbage. */
    status = HAL_I2C_Mem_Read(hi2c, MPU6050_I2C_ADDR, MPU6050_REG_WHO_AM_I,
                               I2C_MEMADD_SIZE_8BIT, &who_am_i, 1, HAL_MAX_DELAY);
    if (status != HAL_OK || who_am_i != MPU6050_WHO_AM_I_EXPECTED) {
        return HAL_ERROR;
    }

    /* 2. Wake the sensor up -- MPU6050 boots with the sleep bit (bit 6)
     *    set in PWR_MGMT_1, so no data updates until it's cleared. */
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_REG_PWR_MGMT_1,
                                I2C_MEMADD_SIZE_8BIT, &zero, 1, HAL_MAX_DELAY);
    if (status != HAL_OK) {
        return status;
    }

    /* 3. Explicitly set the full-scale ranges this driver's scale
     *    constants assume (+/-2g, +/-250 deg/s), rather than relying on
     *    the chip's power-on-reset default matching by coincidence. */
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_REG_ACCEL_CONFIG,
                                I2C_MEMADD_SIZE_8BIT, &zero, 1, HAL_MAX_DELAY);
    if (status != HAL_OK) {
        return status;
    }
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_REG_GYRO_CONFIG,
                                I2C_MEMADD_SIZE_8BIT, &zero, 1, HAL_MAX_DELAY);
    return status;
}

HAL_StatusTypeDef MPU6050_ReadScaled(I2C_HandleTypeDef *hi2c, MPU6050_Reading *out)
{
    uint8_t raw[14];
    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(
        hi2c, MPU6050_I2C_ADDR, MPU6050_REG_ACCEL_XOUT_H,
        I2C_MEMADD_SIZE_8BIT, raw, sizeof(raw), HAL_MAX_DELAY);
    if (status != HAL_OK) {
        return status;
    }

    /* Each axis is a signed 16-bit big-endian value. Byte layout of the
     * 14-byte burst starting at ACCEL_XOUT_H (0x3B):
     *   0-1 accel X, 2-3 accel Y, 4-5 accel Z,
     *   6-7 temperature (skipped -- not used here),
     *   8-9 gyro X, 10-11 gyro Y, 12-13 gyro Z. */
    int16_t ax_raw = (int16_t)((raw[0]  << 8) | raw[1]);
    int16_t ay_raw = (int16_t)((raw[2]  << 8) | raw[3]);
    int16_t az_raw = (int16_t)((raw[4]  << 8) | raw[5]);
    int16_t gx_raw = (int16_t)((raw[8]  << 8) | raw[9]);
    int16_t gy_raw = (int16_t)((raw[10] << 8) | raw[11]);
    int16_t gz_raw = (int16_t)((raw[12] << 8) | raw[13]);

    out->accel_x_g  = ax_raw / MPU6050_ACCEL_SCALE_LSB_PER_G;
    out->accel_y_g  = ay_raw / MPU6050_ACCEL_SCALE_LSB_PER_G;
    out->accel_z_g  = az_raw / MPU6050_ACCEL_SCALE_LSB_PER_G;
    out->gyro_x_dps = gx_raw / MPU6050_GYRO_SCALE_LSB_PER_DPS;
    out->gyro_y_dps = gy_raw / MPU6050_GYRO_SCALE_LSB_PER_DPS;
    out->gyro_z_dps = gz_raw / MPU6050_GYRO_SCALE_LSB_PER_DPS;

    return HAL_OK;
}
